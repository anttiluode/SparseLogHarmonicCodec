import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import soundfile as sf
import sounddevice as sd
import threading
import os
import zlib
import struct
import pickle
import time
import scipy.signal
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. POST-PROCESSOR (Adaptive to Variable SR)
# ==============================================================================
class PostProcessor:
    @staticmethod
    def apply_chain(audio, sr, use_fx=True, crush_sr=44100):
        # 1. ADAPTIVE NOTCH (The Valley)
        if use_fx:
            nyquist = sr / 2
            # Target range: 1300Hz - 6300Hz
            target_low = 1300
            target_high = 6300
            
            # Only apply filter if we have enough bandwidth
            if target_low < (nyquist - 200):
                actual_high = min(target_high, nyquist - 100)
                if actual_high > target_low + 100:
                    low_norm = target_low / nyquist
                    high_norm = actual_high / nyquist
                    sos = scipy.signal.butter(2, [low_norm, high_norm], btype='bandstop', output='sos')
                    audio = scipy.signal.sosfilt(sos, audio)

        # 2. OUTPUT BITCRUSH (Texture)
        # If output crush is LOWER than the engine rate, we crush further.
        if crush_sr < (sr - 100):
            step = int(sr / crush_sr)
            if step > 1:
                len_orig = len(audio)
                downsampled = audio[::step]
                crushed = np.repeat(downsampled, step)
                if len(crushed) > len_orig: audio = crushed[:len_orig]
                else: audio = np.pad(crushed, (0, len_orig - len(crushed)))

        # 3. HYPER CHORUS
        if use_fx:
            len_orig = len(audio)
            xp = np.arange(len_orig)
            slow = np.interp(np.arange(len_orig) * 0.997, xp, audio, left=0, right=0)
            fast = np.interp(np.arange(len_orig) * 1.003, xp, audio, left=0, right=0)
            return (audio * 0.5) + (slow * 0.25) + (fast * 0.25)
        
        return audio

# ==============================================================================
# 2. TINY NEURAL NETWORK (Adaptive)
# ==============================================================================
class TinyAudioNN:
    def __init__(self, input_size, hidden_size=64):
        self.input_size = input_size
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, input_size))
        self.lr = 0.01 
        
    def train_one_epoch(self, inputs, targets):
        batch_size = 32
        n_samples = inputs.shape[0]
        indices = np.arange(n_samples); np.random.shuffle(indices)
        for i in range(0, n_samples, batch_size):
            idx = indices[i:i+batch_size]
            X_b = inputs[idx]; Y_b = targets[idx]
            
            z1 = np.dot(X_b, self.W1) + self.b1; a1 = np.tanh(z1)
            out = np.dot(a1, self.W2) + self.b2
            
            diff = out - Y_b; m = X_b.shape[0]
            dW2 = (1/m) * np.dot(a1.T, diff)
            db2 = (1/m) * np.sum(diff, axis=0, keepdims=True)
            dH = np.dot(diff, self.W2.T) * (1 - a1**2)
            dW1 = (1/m) * np.dot(X_b.T, dH)
            db1 = (1/m) * np.sum(dH, axis=0, keepdims=True)
            
            self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
            self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2

    def get_weights(self):
        return {'W1':self.W1.astype(np.float16), 'b1':self.b1.astype(np.float16),
                'W2':self.W2.astype(np.float16), 'b2':self.b2.astype(np.float16)}
    
    def set_weights(self, w):
        self.W1=w['W1'].astype(np.float32); self.b1=w['b1'].astype(np.float32)
        self.W2=w['W2'].astype(np.float32); self.b2=w['b2'].astype(np.float32)

# ==============================================================================
# 3. TRUE VARIABLE ENGINE (The File Size Fix)
# ==============================================================================
class NeuralVariableEngine:
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        self.nn = None
        self.log_indices = None
        
    def analyze_and_train(self, audio_in, original_sr, target_sr, top_k=32, train_nn=True):
        # 1. PHYSICAL RESAMPLE (Reduce Frame Count)
        if target_sr != original_sr:
            num_samples = int(len(audio_in) * (target_sr / original_sr))
            audio_work = scipy.signal.resample(audio_in, num_samples)
        else:
            audio_work = audio_in

        # 2. Analyze on the LOWER rate
        pad = (self.chunk_size - (len(audio_work) % self.chunk_size)) % self.chunk_size
        padded = np.pad(audio_work, (0, pad))
        chunks = padded.reshape(-1, self.chunk_size)
        
        # Frequencies based on TARGET RATE
        freqs = rfftfreq(self.chunk_size, 1/target_sr)
        self.log_indices = np.unique(np.logspace(0, np.log10(len(freqs)-1), 128).astype(int))
        input_dim = len(self.log_indices)
        
        law_data = []; X_train = []
        
        for i, chunk in enumerate(chunks):
            spectrum = rfft(chunk)
            mag = np.abs(spectrum); phase = np.angle(spectrum)
            shell_mags = mag[self.log_indices]
            
            if train_nn: X_train.append(np.log1p(shell_mags))
            
            if top_k < input_dim:
                top_k_idx = np.argsort(shell_mags)[-top_k:]
                active_indices = self.log_indices[top_k_idx]
            else:
                active_indices = self.log_indices
                
            law_data.append({
                'idx': active_indices.astype(np.uint8),
                'mag': mag[active_indices].astype(np.float16),
                'phs': phase[active_indices].astype(np.float16)
            })

        nn_weights = None
        if train_nn and len(X_train) > 0:
            X_data = np.array(X_train)
            self.nn = TinyAudioNN(input_dim, 64)
            self.nn.train_one_epoch(X_data, X_data)
            nn_weights = self.nn.get_weights()
            
        return law_data, nn_weights, len(padded) # Return new shortened length

    def synthesize(self, law_data, nn_weights, current_sr, current_len):
        chunks = []
        freqs = rfftfreq(self.chunk_size, 1/current_sr)
        
        # Adaptive Indices
        if self.log_indices is None or len(self.log_indices) == 0:
            self.log_indices = np.unique(np.logspace(0, np.log10(len(freqs)-1), 128).astype(int))
            
        if nn_weights:
            dim = nn_weights['W1'].shape[0]
            self.nn = TinyAudioNN(dim, 64)
            self.nn.set_weights(nn_weights)
            
        for chunk_data in law_data:
            spectrum = np.zeros(self.chunk_size//2 + 1, dtype=np.complex64)
            idx = chunk_data['idx']
            mag = chunk_data['mag'].astype(np.float32)
            phs = chunk_data['phs'].astype(np.float32)
            
            real = mag * np.cos(phs); imag = mag * np.sin(phs)
            spectrum[idx] = real + 1j * imag
            chunks.append(irfft(spectrum))
            
        full = np.concatenate(chunks)[:current_len]
        return full

# ==============================================================================
# 4. PACKER (Must save SR)
# ==============================================================================
class VariablePacker:
    @staticmethod
    def pack(law_data, nn_weights, sr, length):
        # We save the VARIABLE SR
        payload = {'sr': sr, 'len': length, 'data': law_data, 'nn': nn_weights}
        raw = pickle.dumps(payload)
        compressed = zlib.compress(raw, level=9)
        return struct.pack('4sI', b'VAR1', len(compressed)) + compressed

    @staticmethod
    def unpack(file_bytes):
        header = file_bytes[:8]
        magic, size = struct.unpack('4sI', header)
        if magic != b'VAR1': raise ValueError("Invalid VAR1 File")
        payload = pickle.loads(zlib.decompress(file_bytes[8:]))
        return payload['data'], payload['nn'], payload['sr'], payload['len']

# ==============================================================================
# 5. GUI
# ==============================================================================
class NeuralVariableStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("HOLO LAW // NEURAL + VARIABLE")
        self.root.geometry("1000x700")
        self.root.configure(bg="#111")
        
        self.engine = NeuralVariableEngine(chunk_size=2048)
        self.raw_audio = None; self.raw_sr = 44100
        
        self.law_audio = None; self.law_sr = 44100 # Variable rate audio
        self.law_data = None; self.nn_weights = None
        self.base_recon = None # <--- FIXED: Initialized to None
        
        self.is_playing = False; self.active_buffer = None; self.active_sr = 44100
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style(); style.theme_use('clam')
        style.configure("TFrame", background="#111"); style.configure("TLabel", background="#111", foreground="#ddd")
        style.configure("TButton", font=('Segoe UI', 8, 'bold'))
        
        top = ttk.Frame(self.root, padding=10); top.pack(fill=tk.X)
        
        # Load
        f_frame = ttk.Frame(top); f_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_frame, text="📂 LOAD", command=self.load_wav).pack(fill=tk.X)
        self.use_nn = tk.BooleanVar(value=True)
        ttk.Checkbutton(f_frame, text="Train NN", variable=self.use_nn).pack()
        ttk.Button(f_frame, text="⚡ EXTRACT", command=self.run_process).pack(fill=tk.X)

        # Sliders (Release Trigger)
        s_frame = ttk.Frame(top); s_frame.pack(side=tk.LEFT, padx=15)
        
        # Top-K
        self.k_var = tk.IntVar(value=32)
        self._mk_slider(s_frame, "Top-K", 8, 128, self.k_var, 
                        lambda: self.lbl_k.config(text=str(int(self.k_var.get()))), self.run_process)
        self.lbl_k = ttk.Label(s_frame, text="32", width=3); self.lbl_k.pack(side=tk.LEFT)

        # TARGET HZ (The File Size Control)
        self.tsr_var = tk.IntVar(value=44100)
        self._mk_slider(s_frame, "TargetHz", 4000, 44100, self.tsr_var, 
                       lambda: self.lbl_tsr.config(text=f"{int(self.tsr_var.get())//1000}k"), self.run_process)
        self.lbl_tsr = ttk.Label(s_frame, text="44k", width=3); self.lbl_tsr.pack(side=tk.LEFT)

        # CRUSH HZ (Texture FX)
        self.crush_var = tk.IntVar(value=44100)
        self._mk_slider(s_frame, "CrushHz", 4000, 44100, self.crush_var, 
                        self.upd_crush_lbl, self.apply_fx_live)
        self.lbl_cr = ttk.Label(s_frame, text="OFF", width=3); self.lbl_cr.pack(side=tk.LEFT)

        self.use_fx = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="HyperFX", variable=self.use_fx, command=self.apply_fx_live).pack(side=tk.LEFT, padx=10)

        io = ttk.Frame(top); io.pack(side=tk.RIGHT)
        ttk.Button(io, text="💾 SAVE", command=self.save_file).pack(fill=tk.X)
        ttk.Button(io, text="📂 OPEN", command=self.load_file).pack(fill=tk.X)
        ttk.Button(io, text="📤 WAV", command=self.save_wav).pack(fill=tk.X)
        
        self.lbl_stat = ttk.Label(self.root, text="Ready", foreground="yellow"); self.lbl_stat.pack(side=tk.TOP)
        
        vis = ttk.Frame(self.root); vis.pack(fill=tk.BOTH, expand=True, padx=10)
        self.fig = plt.Figure(figsize=(8, 4), dpi=100, facecolor="#111")
        self.ax1 = self.fig.add_subplot(211); self.ax2 = self.fig.add_subplot(212)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, vis); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        bot = ttk.Frame(self.root, padding=10); bot.pack(fill=tk.X)
        ttk.Button(bot, text="⏹", width=3, command=lambda: sd.stop()).pack(side=tk.LEFT)
        # Note: Playback logic adapts to SR
        ttk.Button(bot, text="PLAY ORIG", command=lambda: self.play(self.raw_audio, self.raw_sr)).pack(side=tk.LEFT, padx=5)
        ttk.Button(bot, text="PLAY LAW", command=lambda: self.play(self.law_audio, self.law_sr)).pack(side=tk.LEFT, padx=5)
        
        self.timeline = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(bot, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.timeline)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

    def _mk_slider(self, p, txt, mi, ma, var, lbl_cmd, rel_cmd):
        f = ttk.Frame(p); f.pack(side=tk.LEFT, padx=5)
        ttk.Label(f, text=txt, font=('Segoe UI', 7)).pack()
        s = ttk.Scale(f, from_=mi, to=ma, variable=var, orient=tk.HORIZONTAL, length=80)
        s.pack()
        if lbl_cmd: s.configure(command=lambda v: lbl_cmd())
        if rel_cmd: s.bind("<ButtonRelease-1>", lambda e: rel_cmd())

    def upd_crush_lbl(self, _=None):
        v = int(self.crush_var.get())
        self.lbl_cr.config(text="OFF" if v > 43000 else f"{v//1000}k")

    def load_wav(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not p: return
        d, sr = sf.read(p)
        if len(d.shape)>1: d = d.mean(axis=1)
        self.raw_audio = d.astype(np.float32); self.raw_sr = sr
        self.tsr_var.set(sr); self.lbl_tsr.config(text=f"{sr//1000}k")
        self.lbl_stat.config(text=f"Loaded {len(d)/sr:.1f}s")
        self.run_process()

    def run_process(self):
        if self.raw_audio is None: return
        threading.Thread(target=self._run).start()

    def _run(self):
        self.lbl_stat.config(text="Processing...")
        target = self.tsr_var.get()
        
        # 1. VARIABLE ANALYSIS (True Resampling)
        self.law_data, self.nn_weights, length = self.engine.analyze_and_train(
            self.raw_audio, self.raw_sr, target_sr=target, 
            top_k=self.k_var.get(), train_nn=self.use_nn.get()
        )
        self.law_sr = target # Reconstruction rate
        
        # 2. SYNTHESIS (At low rate)
        self.base_recon = self.engine.synthesize(self.law_data, self.nn_weights, self.law_sr, length)
        
        self.root.after(0, self.apply_fx_live)

    def apply_fx_live(self, _=None):
        if self.base_recon is None: return
        
        self.law_audio = PostProcessor.apply_chain(
            self.base_recon, self.law_sr, 
            use_fx=self.use_fx.get(), 
            crush_sr=self.crush_var.get()
        )
        self._plot()

    def _plot(self):
        self.ax1.clear(); self.ax1.plot(self.raw_audio[::100], 'gray', lw=0.5)
        self.ax1.set_title(f"Original ({self.raw_sr}Hz)", color='white', fontsize=8)
        self.ax1.set_facecolor("#050505"); self.ax1.tick_params(colors='white')
        
        self.ax2.clear(); self.ax2.plot(self.law_audio[::100], color='#00ff99', alpha=0.8, lw=0.5)
        self.ax2.set_title(f"Neural Law ({self.law_sr}Hz) + FX", color='white', fontsize=8)
        self.ax2.set_facecolor("#050505"); self.ax2.tick_params(colors='white')
        self.canvas.draw()
        
        # Compression Calc
        orig_bytes = (len(self.raw_audio) * 2) # 16bit approx
        law_bytes = len(self.law_data) * self.k_var.get() * 5
        self.lbl_stat.config(text=f"Ratio: ~{int(orig_bytes/law_bytes)}x")

    def save_file(self):
        if not hasattr(self, 'law_data'): return
        p = filedialog.asksaveasfilename(defaultextension=".law")
        if p:
            d = VariablePacker.pack(self.law_data, self.nn_weights, self.law_sr, len(self.law_audio))
            with open(p, 'wb') as f: f.write(d)
            self.lbl_stat.config(text=f"Saved {len(d)/1024:.1f}KB")

    def load_file(self):
        p = filedialog.askopenfilename(filetypes=[("Holo Law", "*.law")])
        if p:
            with open(p, 'rb') as f: d = f.read()
            ld, nw, sr, l = VariablePacker.unpack(d)
            self.law_sr = sr
            self.base_recon = self.engine.synthesize(ld, nw, sr, l)
            self.raw_audio = None
            self.apply_fx_live()

    def save_wav(self):
        if self.law_audio is None: return
        p = filedialog.asksaveasfilename(defaultextension=".wav")
        if p: sf.write(p, self.law_audio, self.law_sr)

    def play(self, a, sr):
        if a is not None:
            sd.stop(); self.active_buffer=a; self.active_sr=sr
            sd.play(a, sr)
            self.is_playing=True; self.start=time.time(); self.root.after(100, self._loop)

    def _loop(self):
        if self.is_playing:
            pos = (time.time()-self.start)*self.active_sr
            if pos < len(self.active_buffer):
                self.timeline.set((pos/len(self.active_buffer))*100)
                self.root.after(100, self._loop)
            else: self.is_playing=False

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralVariableStudio(root)
    root.mainloop()