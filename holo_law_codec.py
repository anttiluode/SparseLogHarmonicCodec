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
from scipy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ==============================================================================
# 1. THE PHYSICS ENGINE (The Law Extractor)
# ==============================================================================
class HarmonicLawEngine:
    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size
        
    def analyze(self, audio, sr, top_k=32):
        pad = (self.chunk_size - (len(audio) % self.chunk_size)) % self.chunk_size
        padded = np.pad(audio, (0, pad))
        chunks = padded.reshape(-1, self.chunk_size)
        
        # Logarithmic Basis
        freqs = rfftfreq(self.chunk_size, 1/sr)
        log_indices = np.unique(np.logspace(0, np.log10(len(freqs)-1), 128).astype(int))
        
        law_data = []
        ghost_signal = np.zeros_like(padded)
        
        for i, chunk in enumerate(chunks):
            spectrum = rfft(chunk)
            mag = np.abs(spectrum)
            phase = np.angle(spectrum)
            
            shell_mags = mag[log_indices]
            
            if top_k < len(log_indices):
                top_k_local_idx = np.argsort(shell_mags)[-top_k:]
                active_indices = log_indices[top_k_local_idx]
            else:
                active_indices = log_indices
                
            chunk_law = {
                'idx': active_indices.astype(np.uint8),
                'mag': mag[active_indices].astype(np.float16),
                'phs': phase[active_indices].astype(np.float16)
            }
            law_data.append(chunk_law)
            
            mask = np.zeros_like(spectrum, dtype=bool)
            mask[active_indices] = True
            recon_spec = spectrum * mask
            recon_chunk = irfft(recon_spec)
            
            start = i * self.chunk_size
            end = start + self.chunk_size
            ghost_signal[start:end] = chunk - recon_chunk

        return law_data, ghost_signal[:len(audio)], len(padded)

    def synthesize(self, law_data, sr, original_len):
        chunks = []
        for chunk_data in law_data:
            spectrum = np.zeros(self.chunk_size//2 + 1, dtype=np.complex64)
            idx = chunk_data['idx']
            mag = chunk_data['mag']
            phs = chunk_data['phs']
            
            real = mag * np.cos(phs)
            imag = mag * np.sin(phs)
            spectrum[idx] = real + 1j * imag
            chunks.append(irfft(spectrum))
            
        full = np.concatenate(chunks)
        return full[:original_len]

# ==============================================================================
# 2. THE PACKER
# ==============================================================================
class LawPacker:
    @staticmethod
    def pack(law_data, sr, original_len):
        payload = {'sr': sr, 'len': original_len, 'data': law_data}
        raw = pickle.dumps(payload)
        compressed = zlib.compress(raw, level=9)
        header = struct.pack('4sI', b'LAW1', len(raw))
        return header + compressed

    @staticmethod
    def unpack(file_bytes):
        header = file_bytes[:8]
        magic, size = struct.unpack('4sI', header)
        if magic != b'LAW1': raise ValueError("Invalid File")
        raw = zlib.decompress(file_bytes[8:])
        payload = pickle.loads(raw)
        return payload['data'], payload['sr'], payload['len']

# ==============================================================================
# 3. THE STUDIO GUI
# ==============================================================================
class LawCodecStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("HOLO LAW CODEC // The Physics Engine")
        self.root.geometry("1100x750") # REDUCED HEIGHT
        self.root.configure(bg="#111")
        
        self.engine = HarmonicLawEngine(chunk_size=2048)
        self.raw_audio = None
        self.recon_audio = None
        self.ghost_audio = None
        self.sr = 44100
        self.original_len = 0
        
        # Playback State
        self.is_playing = False
        self.playback_start_time = 0
        self.playback_offset_samples = 0
        self.active_buffer = None
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#111")
        style.configure("TLabel", background="#111", foreground="#ddd")
        style.configure("TButton", font=('Segoe UI', 9, 'bold'))
        style.configure("Horizontal.TScale", background="#111")
        
        # --- HEADER ---
        head = ttk.Frame(self.root, padding=10)
        head.pack(fill=tk.X)
        ttk.Label(head, text="HOLO LAW EXTRACTOR", font=('Segoe UI', 16, 'bold'), foreground="#00ff99").pack(side=tk.LEFT)
        self.lbl_stats = ttk.Label(head, text="Ready.", foreground="yellow")
        self.lbl_stats.pack(side=tk.RIGHT)

        # --- CONTROLS ---
        ctrl = ttk.Frame(self.root, padding=5)
        ctrl.pack(fill=tk.X)
        
        ttk.Button(ctrl, text="📂 LOAD AUDIO", command=self.load_audio).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(ctrl, text="Sparsity (Top-K):").pack(side=tk.LEFT, padx=(15,5))
        self.k_var = tk.IntVar(value=32)
        scale = ttk.Scale(ctrl, from_=8, to=128, variable=self.k_var, orient=tk.HORIZONTAL, length=150)
        scale.pack(side=tk.LEFT)
        self.lbl_k = ttk.Label(ctrl, text="32")
        self.lbl_k.pack(side=tk.LEFT, padx=5)
        scale.configure(command=lambda v: self.lbl_k.configure(text=f"{int(float(v))}"))
        
        ttk.Button(ctrl, text="⚡ EXTRACT", command=self.run_analysis).pack(side=tk.LEFT, padx=15)
        ttk.Button(ctrl, text="💾 SAVE", command=self.save_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl, text="📂 LOAD", command=self.load_file).pack(side=tk.LEFT, padx=5)

        # --- VISUALIZATION (COMPACT) ---
        vis = ttk.Frame(self.root)
        vis.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Reduced height from 8 to 5
        self.fig = plt.Figure(figsize=(10, 5), dpi=100, facecolor="#111")
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        
        titles = ["Original", "The Law (Math)", "The Ghost (Residual)"]
        colors = ['gray', '#00ff99', '#ff3366']
        
        for ax, t, c in zip([self.ax1, self.ax2, self.ax3], titles, colors):
            ax.set_facecolor("#050505")
            ax.tick_params(colors='white', labelsize=8)
            ax.set_title(t, color=c, fontsize=9, pad=2)
            for spine in ax.spines.values(): spine.set_color('white')
            
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, vis)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # --- PLAYBACK CONTROLS (ALWAYS VISIBLE) ---
        play_frame = ttk.LabelFrame(self.root, text="Playback Control", padding=5)
        play_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Timeline Slider
        self.timeline_var = tk.DoubleVar(value=0)
        self.slider = ttk.Scale(play_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.timeline_var)
        self.slider.pack(fill=tk.X, pady=2)
        self.slider.bind("<ButtonRelease-1>", self.on_seek_release)
        
        # Buttons Row
        btn_row = ttk.Frame(play_frame)
        btn_row.pack(fill=tk.X, pady=2)
        
        ttk.Button(btn_row, text="⏹ STOP", command=self.stop_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="▶ ORIGINAL", command=lambda: self.play(self.raw_audio)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="▶ LAW (MATH)", command=lambda: self.play(self.recon_audio)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="▶ GHOST", command=lambda: self.play(self.ghost_audio)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="▶ MIX", command=self.play_mix).pack(side=tk.LEFT, padx=5)
        
        # Save Wav
        ttk.Button(btn_row, text="📤 EXPORT WAV", command=self.save_wav).pack(side=tk.RIGHT, padx=5)

    # --- LOGIC ---
    def load_audio(self):
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if not p: return
        try:
            data, sr = sf.read(p)
            if len(data.shape) > 1: data = data.mean(axis=1)
            self.raw_audio = data.astype(np.float32)
            self.sr = sr
            self.original_len = len(self.raw_audio)
            
            self.ax1.clear(); self.ax1.plot(self.raw_audio[::100], 'gray')
            self.ax1.set_title("Original", color='white', fontsize=9)
            self.canvas.draw()
            self.lbl_stats.config(text=f"Loaded {len(data)/sr:.2f}s")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_analysis(self):
        if self.raw_audio is None: return
        threading.Thread(target=self._analyze).start()

    def _analyze(self):
        k = self.k_var.get()
        self.lbl_stats.config(text="Extracting Laws...")
        self.law_data, self.ghost_audio, length = self.engine.analyze(self.raw_audio, self.sr, top_k=k)
        self.recon_audio = self.engine.synthesize(self.law_data, self.sr, length)
        self.root.after(0, self._update_plots)

    def _update_plots(self):
        self.ax2.clear()
        if self.recon_audio is not None:
            self.ax2.plot(self.recon_audio[::100], color='#00ff99', alpha=0.8)
            current_k = len(self.law_data[0]['idx']) if self.law_data else 0
            self.ax2.set_title(f"The Law (Top-{current_k} Log-Shells)", color='#00ff99', fontsize=9)
        
        self.ax3.clear()
        if self.ghost_audio is not None:
            self.ax3.plot(self.ghost_audio[::100], color='#ff3366', alpha=0.6)
            self.ax3.set_title(f"The Ghost (Residual Energy: {np.std(self.ghost_audio):.4f})", color='#ff3366', fontsize=9)
        
        if self.original_len > 0 and self.law_data:
            raw_size = self.original_len * 4
            chunk_count = len(self.law_data)
            current_k = len(self.law_data[0]['idx'])
            law_size = chunk_count * current_k * 5
            ratio = raw_size / law_size if law_size > 0 else 0
            self.lbl_stats.config(text=f"Law Compression: ~{ratio:.1f}x")
            
        self.canvas.draw()

    def save_file(self):
        if not hasattr(self, 'law_data'): return
        p = filedialog.asksaveasfilename(defaultextension=".law")
        if not p: return
        bin_data = LawPacker.pack(self.law_data, self.sr, self.original_len)
        with open(p, 'wb') as f: f.write(bin_data)
        self.lbl_stats.config(text=f"Saved .LAW ({len(bin_data)/1024:.1f} KB)")

    def load_file(self):
        p = filedialog.askopenfilename(filetypes=[("Holo Law", "*.law")])
        if not p: return
        try:
            with open(p, 'rb') as f: bin_data = f.read()
            law_data, sr, length = LawPacker.unpack(bin_data)
            self.law_data = law_data
            self.sr = sr
            self.original_len = length
            self.recon_audio = self.engine.synthesize(law_data, sr, length)
            self.raw_audio = None
            self.ghost_audio = None
            self.ax1.clear()
            self.ax1.text(0.5, 0.5, "(Law File Loaded)", color='white', ha='center')
            self._update_plots()
            self.lbl_stats.config(text="Loaded Law File")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    # --- PLAYBACK LOGIC ---
    def on_seek_release(self, event):
        if self.active_buffer is not None:
            perc = self.timeline_var.get()
            new_sample = int((perc / 100) * len(self.active_buffer))
            was_playing = self.is_playing
            self.stop_audio()
            self.playback_offset_samples = new_sample
            if was_playing:
                # To resume properly, we'd need to know which buffer was active.
                # User can just click Play again.
                pass

    def play(self, audio):
        if audio is None: return
        self.stop_audio()
        self.active_buffer = audio
        
        start_sample = int((self.timeline_var.get() / 100) * len(audio))
        if start_sample >= len(audio) - 1000:
            start_sample = 0
            self.timeline_var.set(0)
            
        sd.play(audio[start_sample:], self.sr)
        self.is_playing = True
        self.playback_start_time = time.time()
        self.playback_offset_samples = start_sample
        
        self.root.after(100, self._playback_loop)

    def play_mix(self):
        if self.recon_audio is not None and self.ghost_audio is not None:
            mix = self.recon_audio + (self.ghost_audio * 0.5)
            self.play(mix)

    def stop_audio(self):
        sd.stop()
        self.is_playing = False

    def _playback_loop(self):
        if not self.is_playing or self.active_buffer is None: return
        
        elapsed = time.time() - self.playback_start_time
        samples_played = int(elapsed * self.sr)
        current_pos = self.playback_offset_samples + samples_played
        
        total_samples = len(self.active_buffer)
        
        if current_pos >= total_samples:
            self.stop_audio()
            self.timeline_var.set(100)
        else:
            perc = (current_pos / total_samples) * 100
            self.timeline_var.set(perc)
            self.root.after(100, self._playback_loop)

    def save_wav(self):
        if self.recon_audio is None: return
        p = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV", "*.wav")])
        if not p: return
        
        to_save = self.recon_audio
        if self.ghost_audio is not None:
            res = messagebox.askyesno("Export", "Do you want to include the 'Ghost' layer in the WAV?")
            if res:
                to_save = self.recon_audio + self.ghost_audio
        
        sf.write(p, to_save, self.sr)
        self.lbl_stats.config(text=f"Exported WAV")

if __name__ == "__main__":
    root = tk.Tk()
    app = LawCodecStudio(root)
    root.mainloop()