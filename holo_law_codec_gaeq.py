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
import random
import json
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. GENETIC ALGORITHM ENGINE
# ==============================================================================
class GeneticEQOptimizer:
    def __init__(self, engine, audio_slice, sr, k_target=32):
        self.engine = engine
        self.audio = audio_slice
        self.sr = sr
        self.k = k_target
        self.pop_size = 20
        self.population = []
        self.generation = 0
        self.best_genome = None
        self.best_score = -999999.0

    def spawn(self, num_bands=32):
        self.population = []
        for _ in range(self.pop_size):
            genome = np.random.rand(num_bands)
            mid = num_bands // 2
            # Bias towards mids initially to speed up voice finding
            genome[mid-4:mid+4] += 0.5 
            self.population.append(np.clip(genome, 0.0, 1.0))

    def evaluate(self, genome):
        law_data, _, padded_len = self.engine.analyze(self.audio, self.sr, top_k=self.k, eq_gains=genome)
        recon = self.engine.synthesize(law_data, self.sr, len(self.audio))
        
        fft_orig = np.abs(rfft(self.audio))
        fft_recon = np.abs(rfft(recon))
        
        if fft_orig.shape != fft_recon.shape: return -999999.0

        freqs = rfftfreq(len(self.audio), 1/self.sr)
        weights = np.ones_like(freqs) * 0.1
        # Heavy weighting for human voice range (300Hz - 3400Hz)
        voice_mask = (freqs > 300) & (freqs < 3400)
        weights[voice_mask] = 3.0 
        
        # We want to minimize difference in the voice band
        error = np.mean(weights * (fft_orig - fft_recon)**2)
        return -error 

    def step(self):
        scores = [self.evaluate(dna) for dna in self.population]
        sorted_idx = np.argsort(scores)[::-1]
        self.best_score = scores[sorted_idx[0]]
        self.best_genome = self.population[sorted_idx[0]].copy()
        
        elites = [self.population[i] for i in sorted_idx[:4]]
        next_gen = elites[:]
        while len(next_gen) < self.pop_size:
            p1, p2 = random.choice(elites), random.choice(elites)
            cut = random.randint(0, len(p1))
            child = np.concatenate([p1[:cut], p2[cut:]])
            if random.random() < 0.3: child[random.randint(0, len(child)-1)] += np.random.normal(0, 0.2)
            if random.random() < 0.2: child = np.convolve(child, [0.1, 0.8, 0.1], mode='same')
            next_gen.append(np.clip(child, 0.0, 1.0))
        self.population = next_gen
        self.generation += 1
        return self.best_genome, self.best_score

# ==============================================================================
# 2. PHYSICS ENGINE (Law Only)
# ==============================================================================
class HarmonicLawEngine:
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        
    def analyze(self, audio, sr, top_k=32, eq_gains=None):
        pad = (self.chunk_size - (len(audio) % self.chunk_size)) % self.chunk_size
        padded = np.pad(audio, (0, pad))
        chunks = padded.reshape(-1, self.chunk_size)
        
        freqs = rfftfreq(self.chunk_size, 1/sr)
        log_indices = np.unique(np.logspace(0, np.log10(len(freqs)-1), 128).astype(int))
        
        eq_curve = None
        if eq_gains is not None:
            x_in = np.linspace(0, 1, len(eq_gains))
            x_out = np.linspace(0, 1, len(log_indices))
            eq_curve = np.interp(x_out, x_in, eq_gains)
        
        law_data = []
        
        for i, chunk in enumerate(chunks):
            spectrum = rfft(chunk)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)
            shell_mags = mag[log_indices]
            
            selection_mags = shell_mags * eq_curve if eq_curve is not None else shell_mags
            
            if top_k < len(log_indices):
                top_k_local_idx = np.argsort(selection_mags)[-top_k:]
                active_indices = log_indices[top_k_local_idx]
            else:
                active_indices = log_indices
                
            law_data.append({
                'idx': active_indices.astype(np.uint8),
                'mag': mag[active_indices].astype(np.float16),
                'phs': phase[active_indices].astype(np.float16)
            })

        return law_data, None, len(padded)

    def synthesize(self, law_data, sr, original_len):
        chunks = []
        for chunk_data in law_data:
            spectrum = np.zeros(self.chunk_size//2 + 1, dtype=np.complex64)
            idx = chunk_data['idx']
            mag = chunk_data['mag']
            phs = chunk_data['phs']
            spectrum[idx] = mag * np.cos(phs) + 1j * mag * np.sin(phs)
            chunks.append(irfft(spectrum))
        full = np.concatenate(chunks)
        return full[:original_len]

# ==============================================================================
# 3. LAW PACKER
# ==============================================================================
class LawPacker:
    @staticmethod
    def pack(law_data, sr, original_len):
        payload = {'sr': sr, 'len': original_len, 'data': law_data}
        raw = pickle.dumps(payload)
        compressed = zlib.compress(raw, level=9)
        return struct.pack('4sI', b'LAW1', len(compressed)) + compressed

    @staticmethod
    def unpack(file_bytes):
        header = file_bytes[:8]
        magic, size = struct.unpack('4sI', header)
        if magic != b'LAW1': raise ValueError("Invalid LAW file.")
        payload = pickle.loads(zlib.decompress(file_bytes[8:]))
        return payload['data'], payload['sr'], payload['len']

# ==============================================================================
# 4. GUI & APP
# ==============================================================================
class GraphicalEQ(tk.Frame):
    def __init__(self, parent, width=400, height=60, num_bands=32, callback=None):
        super().__init__(parent)
        self.width, self.height, self.num_bands, self.callback = width, height, num_bands, callback
        self.canvas = tk.Canvas(self, width=width, height=height, bg='#222', highlightthickness=0)
        self.canvas.pack()
        self.gains = np.ones(num_bands)
        self.band_width = width / num_bands
        self.canvas.bind("<B1-Motion>", self._on_drag); self.canvas.bind("<ButtonPress-1>", self._on_click)
        self.draw()

    def _on_click(self, e): self._update(e.x, e.y)
    def _on_drag(self, e): self._update(e.x, e.y)
    def _update(self, x, y):
        idx = int(x // self.band_width)
        if 0 <= idx < self.num_bands:
            self.gains[idx] = np.clip(1.0 - (y / self.height), 0.0, 1.0)
            self.draw()
            if self.callback: self.callback()

    def draw(self):
        self.canvas.delete("all")
        for i, g in enumerate(self.gains):
            x0, y0 = i * self.band_width, self.height * (1 - g)
            self.canvas.create_rectangle(x0, y0, x0 + self.band_width - 1, self.height, fill="#00ff99" if g > 0.5 else "#008855", outline="")

class SighLawStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("SIGH LAW CODEC // GEOMETRIC AUDIO")
        self.root.geometry("1100x750")
        self.root.configure(bg="#111")
        
        self.engine = HarmonicLawEngine(chunk_size=2048)
        self.ga_engine = None
        self.raw_audio = None
        self.recon_audio = None
        self.sr = 44100
        
        self.is_playing = False
        self.is_evolving = False
        self.active_buffer = None
        self._update_job = None
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style(); style.theme_use('clam')
        style.configure("TFrame", background="#111"); style.configure("TLabel", background="#111", foreground="#ddd")
        style.configure("TButton", font=('Segoe UI', 8, 'bold'))
        
        top = ttk.Frame(self.root, padding=5); top.pack(fill=tk.X)
        
        # File & Controls
        f_frame = ttk.LabelFrame(top, text="Core", padding=2); f_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(f_frame, text="📂 LOAD WAV", command=self.load_wav).pack(fill=tk.X, pady=1)
        
        s_frame = ttk.Frame(f_frame); s_frame.pack(fill=tk.X, pady=2)
        self.k_var = tk.IntVar(value=32)
        ttk.Label(s_frame, text="K:").pack(side=tk.LEFT)
        scale = ttk.Scale(s_frame, from_=8, to=128, variable=self.k_var, orient=tk.HORIZONTAL, length=60)
        scale.pack(side=tk.LEFT)
        self.lbl_k = ttk.Label(s_frame, text="32"); self.lbl_k.pack(side=tk.LEFT)
        scale.configure(command=lambda v: [self.lbl_k.configure(text=f"{int(float(v))}"), self.on_param_change()])

        # EQ & GA
        eq_frame = ttk.LabelFrame(top, text="Sigh EQ (Guide the Law)", padding=2); eq_frame.pack(side=tk.LEFT, padx=10)
        self.eq = GraphicalEQ(eq_frame, width=350, height=60, num_bands=32, callback=self.on_param_change)
        self.eq.pack()
        
        ga_frame = ttk.Frame(eq_frame); ga_frame.pack(fill=tk.X, pady=2)
        self.btn_evolve = ttk.Button(ga_frame, text="🧬 AUTO-EQ (Find Voice)", command=self.toggle_evolution)
        self.btn_evolve.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        ttk.Button(ga_frame, text="💾 SAVE EQ", width=8, command=self.save_eq_env).pack(side=tk.LEFT)
        ttk.Button(ga_frame, text="📂 LOAD EQ", width=8, command=self.load_eq_env).pack(side=tk.LEFT)
        self.lbl_gen = ttk.Label(ga_frame, text="Gen: 0", font=('Courier', 8)); self.lbl_gen.pack(side=tk.RIGHT)

        # Playback & IO
        right = ttk.Frame(top); right.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        play_f = ttk.LabelFrame(right, text="Playback", padding=2); play_f.pack(fill=tk.X)
        self.timeline = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(play_f, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.timeline, length=200)
        self.slider.pack(fill=tk.X); self.slider.bind("<ButtonRelease-1>", self.on_seek)
        
        b_row = ttk.Frame(play_f); b_row.pack(fill=tk.X)
        ttk.Button(b_row, text="⏹", width=3, command=self.stop).pack(side=tk.LEFT)
        ttk.Button(b_row, text="ORIGINAL", width=8, command=lambda: self.play(self.raw_audio)).pack(side=tk.LEFT)
        ttk.Button(b_row, text="LAW RECON", width=10, command=lambda: self.play(self.recon_audio)).pack(side=tk.LEFT)
            
        io_row = ttk.Frame(right); io_row.pack(fill=tk.X, pady=5)
        ttk.Button(io_row, text="💾 SAVE LAW", command=self.save_law).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_row, text="📂 LOAD LAW", command=self.load_law).pack(side=tk.LEFT, padx=2)
        ttk.Button(io_row, text="📤 WAV", width=6, command=self.save_wav).pack(side=tk.RIGHT, padx=2)

        self.lbl_stats = ttk.Label(top, text="Ready.", foreground="yellow"); self.lbl_stats.pack(side=tk.RIGHT, padx=10)

        # Viz
        vis = ttk.Frame(self.root); vis.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig = plt.Figure(figsize=(8, 4), dpi=100, facecolor="#111")
        self.ax1 = self.fig.add_subplot(211); self.ax2 = self.fig.add_subplot(212)
        for ax, t, c in zip([self.ax1, self.ax2], ["Original", "The Law"], ['gray', '#00ff99']):
            ax.set_facecolor("#050505"); ax.tick_params(colors='white', labelsize=7)
            ax.set_title(t, color=c, fontsize=9, pad=2); ax.spines['bottom'].set_color('white')
            ax.spines['left'].set_color('white')
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, vis); self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- CORE ---
    def load_wav(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not p: return
        data, sr = sf.read(p)
        if len(data.shape) > 1: data = data.mean(axis=1)
        self.raw_audio = data.astype(np.float32); self.sr = sr
        self.original_len = len(data)
        self.ax1.clear(); self.ax1.plot(self.raw_audio[::100], 'gray', lw=0.5)
        self.ax1.set_title("Original", color='white', fontsize=9); self.canvas.draw()
        self.lbl_stats.config(text=f"Loaded {len(data)/sr:.1f}s")
        self.eq.gains[:] = 1.0; self.eq.draw()
        self._perform_analysis() # Initial run

    def on_param_change(self):
        if self._update_job: self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(300, self.run_analysis)

    def run_analysis(self):
        if self.raw_audio is None: return
        # Prevent manual changes from interfering if GA is running
        if self.is_evolving: return 
        if threading.active_count() > 3: return 
        threading.Thread(target=self._perform_analysis).start()

    def _perform_analysis(self):
        # This is the SHARED method used by both Manual and Auto modes
        # It reads the current EQ state and updates the Law
        if self.raw_audio is None: return
        
        # Get current gains (thread-safe enough for GUI read)
        gains = self.eq.gains.copy()
        
        self.law_data, _, length = self.engine.analyze(self.raw_audio, self.sr, top_k=self.k_var.get(), eq_gains=gains)
        self.recon_audio = self.engine.synthesize(self.law_data, self.sr, length)
        
        # Update Plots (on main thread)
        self.root.after(0, self._update_plots)

    def _update_plots(self):
        self.ax2.clear(); self.ax2.plot(self.recon_audio[::100], color='#00ff99', alpha=0.8, lw=0.5)
        self.ax2.set_title(f"The Law (Top-{self.k_var.get()} Guided)", color='#00ff99', fontsize=9)
        self.canvas.draw()
        if not self.is_evolving: self.lbl_stats.config(text="Law Updated.")

    def toggle_evolution(self):
        if self.is_evolving:
            self.is_evolving = False
            self.btn_evolve.config(text="🧬 AUTO-EQ")
        else:
            if self.raw_audio is None: return
            self.is_evolving = True
            self.btn_evolve.config(text="⏹ STOP")
            threading.Thread(target=self._evolve_loop, daemon=True).start()

    def _evolve_loop(self):
        slice_len = 2 * self.sr 
        if len(self.raw_audio) > slice_len:
            rms = [np.mean(self.raw_audio[i:i+slice_len]**2) for i in range(0, len(self.raw_audio)-slice_len, self.sr)]
            best_start = np.argmax(rms) * self.sr
            chunk = self.raw_audio[best_start:best_start+slice_len]
        else:
            chunk = self.raw_audio

        self.ga_engine = GeneticEQOptimizer(self.engine, chunk, self.sr, k_target=self.k_var.get())
        self.ga_engine.spawn(num_bands=32)
        
        while self.is_evolving:
            best_dna, score = self.ga_engine.step()
            # Update GUI live every 5 generations to be snappy
            if self.ga_engine.generation % 5 == 0:
                self.root.after(0, lambda d=best_dna, s=score, g=self.ga_engine.generation: self._update_ga_ui(d, s, g))
            time.sleep(0.01)

    def _update_ga_ui(self, dna, score, gen):
        # 1. Update EQ Visuals
        self.eq.gains = dna
        self.eq.draw()
        self.lbl_gen.config(text=f"Gen: {gen} | Fit: {score:.2f}")
        
        # 2. CRITICAL FIX: Trigger the analysis logic to update audio
        # We spawn a thread to avoid blocking the UI update, bypassing the 'is_evolving' lock
        threading.Thread(target=self._perform_analysis).start()

    def save_law(self):
        if not hasattr(self, 'law_data'): return
        p = filedialog.asksaveasfilename(defaultextension=".law", filetypes=[("Holo Law", "*.law")])
        if p:
            data = LawPacker.pack(self.law_data, self.sr, self.original_len)
            with open(p, 'wb') as f: f.write(data)
            self.lbl_stats.config(text=f"Saved Law ({len(data)/1024:.1f} KB)")

    def load_law(self):
        p = filedialog.askopenfilename(filetypes=[("Holo Law", "*.law")])
        if p:
            with open(p, 'rb') as f: d = f.read()
            self.law_data, self.sr, self.original_len = LawPacker.unpack(d)
            self.recon_audio = self.engine.synthesize(self.law_data, self.sr, self.original_len)
            self.raw_audio = None
            
            self.eq.gains[:] = 1.0; self.eq.draw() # Reset EQ on load
            self.ax1.clear(); self.ax1.text(0.5,0.5,"(Law Loaded)", color='white', ha='center')
            self._update_plots()

    def save_eq_env(self):
        p = filedialog.asksaveasfilename(defaultextension=".env", filetypes=[("EQ Envelope", "*.env")])
        if p: np.save(p, self.eq.gains); self.lbl_stats.config(text="Saved EQ")

    def load_eq_env(self):
        p = filedialog.askopenfilename(filetypes=[("EQ Envelope", "*.env")])
        if p:
            self.eq.gains = np.load(p); self.eq.draw()
            if self.raw_audio is not None: self.run_analysis()

    def save_wav(self):
        if self.recon_audio is None: return
        p = filedialog.asksaveasfilename(defaultextension=".wav")
        if p: sf.write(p, self.recon_audio, self.sr)

    def on_seek(self, e): pass
    def play(self, audio):
        if audio is None: return
        sd.stop(); self.active_buffer = audio
        sd.play(audio, self.sr)
        self.is_playing = True
        self.playback_start_time = time.time()
        self.root.after(100, self._pb_loop)
    def stop(self): sd.stop(); self.is_playing = False
    def _pb_loop(self):
        if self.is_playing:
            e = time.time() - self.playback_start_time
            self.timeline.set(min(100, (e / (len(self.active_buffer)/self.sr))*100))
            if e < (len(self.active_buffer)/self.sr): self.root.after(100, self._pb_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = SighLawStudio(root)
    root.mainloop()