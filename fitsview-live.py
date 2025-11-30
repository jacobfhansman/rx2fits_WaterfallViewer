#!/usr/bin/env python3
from matplotlib import use
use("TkAgg")  # Let tkinter talk to matplotlib

import socket, threading, time, collections, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.gridspec import GridSpec
import datetime as dt

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    CTK = True  # All "if CTK" checks relating to whether to put customtkinter or regular UI up
except Exception:
    import tkinter as ctk
    import tkinter.ttk as ttk
    CTK = False

import Typinski

host, port = "127.0.0.1", 5555

log_scale = 65536.0 / 4.8165       # ~= 13606.9, Map log10(65536) = 4.8165 to 65536

np_dtype_by_bitpix = {  # fits is big-endian and signed, have all types available and pull correct one from header
    -64: ">f8",   # float64
    -32: ">f4",   # float32
     16: ">i2",   # signed 16-bit
     32: ">i4",   # signed 32-bit
}

unipolar = Typinski.Typinski
bipolar  = plt.get_cmap("seismic")

sao = colors.LinearSegmentedColormap(
    "sao",
    {
        "green": [(0.0,0.0,0.0),(0.2,0.0,0.0),(0.4,1.0,1.0),(0.8,1.0,1.0),(1.0,0.0,0.0)],
        "red":   [(0.0,1.0,1.0),(0.2,0.0,0.0),(0.6,0.0,0.0),(0.8,1.0,1.0),(1.0,1.0,1.0)],
        "blue":  [(0.0,1.0,1.0),(0.4,1.0,1.0),(0.6,0.0,0.0),(1.0,0.0,0.0)],
    }, 1024
)

unk = colors.LinearSegmentedColormap(
    "unk",
    {
        "red":   [(a/255, np.abs(1 - np.cos(3*np.pi*(a/255)**2)), np.abs(1 - np.cos(3*np.pi*(a/255)**2))) for a in range(256)],
        "green": [(a/255, np.abs(np.sin(1.5*np.pi*(a/255)))**2, np.abs(np.sin(1.5*np.pi*(a/255)))**2) for a in range(256)],
        "blue":  [(a/255, np.sin(1.5*np.pi*(a/255)**0.5)**2, np.sin(1.5*np.pi*(a/255)**0.5)**2) for a in range(256)],
    }, 1024
)

aj4co = colors.LinearSegmentedColormap(
    "aj4co",
    {
        "red":   [(0.00,0.00,0.00),(0.10,0.00,0.00),(0.30,0.00,0.00),(0.50,0.20,0.20),(0.70,0.90,0.90),(0.85,1.00,1.00),(1.00,1.00,1.00)],
        "green": [(0.00,0.00,0.00),(0.10,0.00,0.00),(0.30,0.30,0.30),(0.50,0.80,0.80),(0.70,0.70,0.70),(0.85,0.30,0.30),(1.00,1.00,1.00)],
        "blue":  [(0.00,0.00,0.00),(0.10,0.25,0.25),(0.30,0.80,0.80),(0.50,1.00,1.00),(0.70,0.20,0.20),(0.85,0.00,0.00),(1.00,1.00,1.00)],
    }, 1024
)

def get_cmap(name):  # get colormap from the dropdown
    if name == "unipolar": return unipolar
    if name == "aj4co":    return aj4co
    if name == "sao":      return sao
    if name == "unk":      return unk
    if name == "bipolar":  return bipolar

def recv_exact(sock, n):  # Add to the buffer when socket open
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def read_fits_header(conn):  # Get da heada
    header = b""
    while True:
        block = recv_exact(conn, 2880)
        header += block
        if b"END" in header:
            pad = (2880 - (len(header) % 2880)) % 2880
            if pad:
                header += recv_exact(conn, pad)
            break
    return header

def parse_cards(header_bytes):  # Parse da heada
    txt = header_bytes.decode("ascii", errors="ignore")
    cards = {}
    for i in range(0, len(txt), 80):
        c = txt[i:i+80]
        k = c[:8].strip()
        if k: cards[k] = c
        if k == "END": break
    def ival(k):
        c = cards.get(k, "")
        try: return int(c[10:30].strip())
        except: return None
    def fval(k):
        c = cards.get(k, "")
        try: return float(c[10:30].strip())
        except: return None
    return {"NAXIS1": ival("NAXIS1"), "BITPIX": ival("BITPIX"),
            "CRVAL1": fval("CRVAL1"), "CDELT1": fval("CDELT1"),
            "CRPIX1": fval("CRPIX1"), "BSCALE": fval("BSCALE"), 
            "BZERO": fval("BZERO"), "FREQLOW": fval("FREQLOW"), 
            "FREQHIGH": fval("FREQHIGH"),}

def load_noise_floor(path="median.dat"):  # Get the noise floor output file from Dr. Gray's script
    if not os.path.isfile(path): return None
    try:
        a = np.loadtxt(path, dtype=np.float32).ravel()
        return a if a.size else None
    except:
        return None

class SharedState:
    # Gotta do this because tkinter blocks threads
    # Doing this keeps the data thread / stream connection from being affected by changing the user controls
    def __init__(self):
        self.lock = threading.Lock()
        self.cmap = "unipolar"
        self.use_db = False
        self.interp_on = True
        self.use_nf = load_noise_floor("median.dat") is not None
        self.vmin = -2000.0
        self.vmax = 65536.0
        self.gain = 1.0
        self.offset = 0.0
        self.log_floor_db = -120.0
    def update(self, **kw):
        with self.lock:
            for k, v in kw.items(): setattr(self, k, v)
    def snapshot(self):
        with self.lock:
            return {
                "cmap": self.cmap,
                "use_db": self.use_db,
                "interp_on": self.interp_on,
                "use_nf": self.use_nf,
                "vmin": self.vmin,
                "vmax": self.vmax,
                "gain": self.gain,
                "offset": self.offset
            }

class App:
    #Init, UI layout
    def __init__(self):
        self.state = SharedState()

        self.root = ctk.CTk() if CTK else ctk.Tk()  # use customtkinter if package installed
        self.root.title("rx2fits — Live Waterfall Viewer")
        if CTK: self.root.geometry("1450x860")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)

        self.fig = plt.Figure(figsize=(11.5, 7.8), dpi=100)
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[42, 2], figure=self.fig)
        self.ax  = self.fig.add_subplot(gs[0,0])
        self.cax = self.fig.add_subplot(gs[0,1])

        self.ax.set_xlabel("UTC time")
        self.ax.set_ylabel("Frequency [MHz]")
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)

        if CTK:  # all "if CTK" checks to apply customtkinter or regular UI
            self.ctrl_outer = ctk.CTkFrame(self.root, width=300)  # frame for control bar
        else:
            self.ctrl_outer = ctk.Frame(self.root)
            self.ctrl_canvas = ctk.Canvas(self.ctrl_outer, width=300, highlightthickness=0)
            self.ctrl_scroll = ttk.Scrollbar(self.ctrl_outer, orient="vertical", command=self.ctrl_canvas.yview)
            self.ctrl_inner = ctk.Frame(self.ctrl_canvas)
            self.ctrl_inner.bind("<Configure>", lambda e: self.ctrl_canvas.configure(scrollregion=self.ctrl_canvas.bbox("all")))
            self.ctrl_canvas.create_window((0,0), window=self.ctrl_inner, anchor="nw")
            self.ctrl_canvas.configure(yscrollcommand=self.ctrl_scroll.set)
        self.ctrl_outer.grid(row=0, column=1, sticky="ns", padx=(4,8), pady=8)

        container = self.ctrl_outer if CTK else self.ctrl_inner

        if CTK:
            lbl = ctk.CTkLabel(container, text="color map")
            self.cmap_combo = ctk.CTkComboBox(container, values=["unipolar","aj4co","sao","unk","bipolar"], state="readonly", command=lambda _value=None: self.cb_cmap())
            self.reset_btn  = ctk.CTkButton(container, text="Reset to defaults", command=self.on_reset)
        else:
            lbl = ctk.Label(container, text="color map")
            self.cmap_combo = ttk.Combobox(container, values=["unipolar","aj4co","sao","unk","bipolar"], state="readonly", width=14)
            self.reset_btn  = ctk.Button(container, text="Reset to defaults", command=self.on_reset)
        lbl.grid(row=0, column=0, sticky="ew", pady=(6,2), padx=6)
        self.cmap_combo.set("unipolar")
        if not CTK:
            self.cmap_combo.bind("<<ComboboxSelected>>", self.cb_cmap)  # customtkinter uses lambda so only bind reg. tkinter
        self.cmap_combo.grid(row=1, column=0, sticky="ew", padx=6, pady=(0,10))
        self.reset_btn.grid(row=2, column=0, sticky="ew", padx=6, pady=(0,12))

        if CTK:
            self.chk_db  = ctk.CTkCheckBox(container, text="Log Scaling", command=self.cb_db)
            self.chk_int = ctk.CTkCheckBox(container, text="Interpolate", command=self.cb_int)
            self.chk_nf  = ctk.CTkCheckBox(container, text="- noise floor", command=self.cb_nf)
        else:
            self.db_var  = ctk.BooleanVar(value=False)
            self.int_var = ctk.BooleanVar(value=True)
            self.nf_var  = ctk.BooleanVar(value=self.state.use_nf)
            self.chk_db  = ctk.Checkbutton(container, text="Log Scaling", variable=self.db_var, command=self.cb_db)
            self.chk_int = ctk.Checkbutton(container, text="Interpolate", variable=self.int_var, command=self.cb_int)
            self.chk_nf  = ctk.Checkbutton(container, text="- noise floor", variable=self.nf_var, command=self.cb_nf)

        self.chk_db.grid(row=3, column=0, sticky="w", padx=6, pady=(0,4))
        if CTK: self.chk_int.select()
        self.chk_int.grid(row=4, column=0, sticky="w", padx=6, pady=(0,6))
        if CTK and self.state.use_nf: self.chk_nf.select()
        self.chk_nf.grid(row=5, column=0, sticky="w", padx=6, pady=(0,6))

        self.controls = {}
        def add_slider(label, row, vmin, vmax, init, step=None, is_int=False):
            if CTK:
                lab = ctk.CTkLabel(container, text=label)
                sld = ctk.CTkSlider(container, from_=vmin, to=vmax, number_of_steps=(int((vmax-vmin)/step) if step else None))
                ent = ctk.CTkEntry(container)
            else:
                lab = ctk.Label(container, text=label)
                sld = ctk.Scale(container, from_=vmin, to=vmax, orient="horizontal", resolution=(step if step else 1e-6), length=220)
                ent = ctk.Entry(container)
            lab.grid(row=row, column=0, sticky="w", padx=6)
            sld.grid(row=row+1, column=0, sticky="ew", padx=6)
            ent.grid(row=row+2, column=0, sticky="ew", padx=6, pady=(0,10))
            sld.set(init)
            ent.delete(0, ctk.END); ent.insert(0, str(int(init) if is_int else init))
            self.controls[label] = (sld, ent, vmin, vmax, is_int)

        add_slider("vmin",  10,  -50000.0, 130000.0, -2000.0, None, False)
        add_slider("vmax",  13,  -50000.0, 130000.0,  65536.0, None, False)
        add_slider("gain",  16,      0.10,    10.00,      1.00, 0.01, False)
        add_slider("offset",19,  -20000.00, 20000.00,      0.00, 1.0,  False)

        def wire_slider(name, setter):  # connect the added sliders to value chosen by user
            sld, ent, lo, hi, is_int = self.controls[name]
            def sld_cb(val=None):
                v = sld.get() if CTK else float(sld.get())
                if is_int: v = int(round(v))
                ent.delete(0, ctk.END); ent.insert(0, str(v))
                setter(v)
            def ent_cb(_evt=None):
                txt = ent.get().strip()
                try:
                    v = int(float(txt)) if is_int else float(txt)
                except:
                    v = sld.get() if CTK else float(sld.get())
                v = max(lo, min(hi, v))
                if is_int: v = int(v)
                sld.set(v); setter(v)
            sld.configure(command=lambda v: sld_cb(v))
            ent.bind("<Return>", ent_cb)
            ent.bind("<FocusOut>", ent_cb)

        wire_slider("vmin",   lambda v: self.state.update(vmin=float(v)))
        wire_slider("vmax",   lambda v: self.state.update(vmax=float(v)))
        wire_slider("gain",   lambda v: self.state.update(gain=float(v)))
        wire_slider("offset", lambda v: self.state.update(offset=float(v)))

        if not CTK:
            self.ctrl_outer.grid_columnconfigure(0, weight=1)
            self.ctrl_outer.grid_rowconfigure(0, weight=1)
            self.ctrl_canvas.grid(row=0, column=0, sticky="ns")
            self.ctrl_scroll.grid(row=0, column=1, sticky="ns")

        #set initial data and plotting
        self.nf_raw = load_noise_floor("median.dat")  # cached noise-floor
        self.nf_proc = None
        self._nf_params = None

        self.col_queue = collections.deque(maxlen=1200)
        self.t_queue   = collections.deque(maxlen=1200)
        self.stop_flag = threading.Event()
        self.reset_flag = threading.Event()

        self.wf = None
        self.tvec = []
        self.im = None
        self.cbar = None
        self.last_draw = 0.0
        self.naxis1 = None
        self.row_bytes = None
        self.freq = None
        self.bscale = 1.0
        self.bzero  = 0.0

        self.start_server()
        self.root.after(30, self.on_timer)

    def cb_cmap(self, _evt=None):  # get color map from selection
        name = self.cmap_combo.get().strip()
        self.state.update(cmap=name)

    def cb_db(self):  # set slider ranges and db scale if Log Scale checkbox selected
        val = (self.chk_db.get() if CTK else bool(self.db_var.get()))
        self.state.update(use_db=bool(val))
        if val:
            self.set_slider_range("vmin", -10000, 65536.0, keep=False)
            self.set_slider_range("vmax", -10000, 65536.0, keep=False)
            self.set_value("vmin", 13606.617)
            self.set_value("vmax", 65536)
            self.state.update(vmin=13606.617, vmax=65536)
        else:
            self.set_slider_range("vmin", -50000.0, 130000.0, keep=False) # Back to linear ranges
            self.set_slider_range("vmax", -50000.0, 130000.0, keep=False)
            self.set_value("vmin", -2000.0)
            self.set_value("vmax", 65536.0)
            self.state.update(vmin=-2000.0, vmax=65536.0)

    def cb_int(self):  # interpolate checkbox
        val = (self.chk_int.get() if CTK else bool(self.int_var.get()))
        self.state.update(interp_on=bool(val))

    def cb_nf(self):  # Subtract noise floor checkbox
        val = (self.chk_nf.get() if CTK else bool(self.nf_var.get()))
        self.state.update(use_nf=bool(val))

    def on_reset(self):  # Reset everything button
        if CTK:
            self.chk_db.deselect()
            self.chk_int.select()
            if self.nf_raw is not None: self.chk_nf.select()
            else: self.chk_nf.deselect()
        else:
            self.db_var.set(False); self.int_var.set(True)
            self.nf_var.set(self.nf_raw is not None)
        self.cmap_combo.set("unipolar")
        defaults = {
            "vmin":-2000.0, "vmax":65536.0, "gain":1.0, "offset":0.0
        }
        for k, v in defaults.items(): self.set_value(k, v)

        self._invalidate_nf_proc()  # resets noise floor

        self.state.update(
            cmap="unipolar", use_db=False, interp_on=True,
            use_nf=(self.nf_raw is not None), vmin=-2000.0, vmax=65536.0,
            gain=1.0, offset=0.0, log_floor_db=-120.0
        )
        self.reset_flag.set()

    def set_value(self, name, value):  #set user chosen values from sliders
        sld, ent, lo, hi, is_int = self.controls[name]
        v = int(value) if is_int else float(value)
        sld.set(v)
        ent.delete(0, ctk.END); ent.insert(0, str(v))

    def set_slider_range(self, name, lo, hi, keep=False):
        sld, ent, _, _, is_int = self.controls[name]
        if not CTK:
            sld.configure(from_=lo, to=hi)
        current = float(sld.get())
        if not keep or current < lo or current > hi:
            current = max(lo, min(hi, current))
            sld.set(int(current) if is_int else current)
        ent.delete(0, ctk.END); ent.insert(0, str(int(current) if is_int else current))

    #Local server loop
    def start_server(self):  # make connection
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(1)
        threading.Thread(target=self.accept_loop, daemon=True).start()
        print(f"[fitsview-live] Listening on {host}:{port}")

    def accept_loop(self):  # accept connection
        while True:
            conn, addr = self.srv.accept()
            print("[fitsview-live] Connected from", addr)
            try:
                self.handle_connection(conn)
            except Exception as e:
                print("[fitsview-live] Error:", e)
            finally:
                try: conn.close()
                except: pass
                print("[fitsview-live] Waiting for next connection...")

    #Per-connection init, reader thread
    def handle_connection(self, conn):  # continue connection after header applied
        header = read_fits_header(conn)
        info = parse_cards(header)

        self.naxis1 = info["NAXIS1"]; bitpix = info["BITPIX"]
        bps = abs(bitpix)//8
        self.row_bytes = self.naxis1 * bps

        # signed big-endian dtype!!!!!!!
        self.np_dtype  = np_dtype_by_bitpix.get(bitpix, ">f4")

        self.bscale = info.get("BSCALE", 1.0) if info.get("BSCALE", None) is not None else 1.0  # BSCALE/BZERO from header
        self.bzero  = info.get("BZERO",  0.0) if info.get("BZERO",  None) is not None else 0.0

        flow  = info.get("FREQLOW")
        fhigh = info.get("FREQHIGH")
        self.freq = np.linspace(flow, fhigh, self.naxis1, dtype=np.float32)

        self.ax.set_title("RX888 MKII Output")

        # reset per-connection state or else doom
        self.im = None
        self.cbar = None
        self.wf = None
        self.tvec = []
        self.col_queue.clear()
        self.t_queue.clear()
        self.stop_flag.clear()
        self.reset_flag.clear()
        self._invalidate_nf_proc()

        threading.Thread(target=self.reader_thread, args=(conn,), daemon=True).start()

        while not self.stop_flag.is_set():
            time.sleep(0.05)

    def reader_thread(self, conn):  # manipulate data with user settings
        try:
            while not self.stop_flag.is_set():
                row = recv_exact(conn, self.row_bytes)

                raw = np.frombuffer(row, dtype=self.np_dtype, count=self.naxis1)  # big endian, signed!!!!
                spec = raw.astype(np.float32)

                if (self.bscale != 1.0) or (self.bzero != 0.0):  #Apply BSCALE/BZERO
                    spec = spec * float(self.bscale) + float(self.bzero)

                s = self.state.snapshot()

                if s["use_nf"]:  #Noise floor subtraction
                    self._prepare_nf(spec_len=spec.shape[0])
                    if self.nf_proc is not None and self.nf_proc.shape[0] == spec.shape[0]:
                        spec = np.maximum(spec - self.nf_proc, 0.0)

                #Optional gain/offset
                if s["gain"] != 1.0 or s["offset"] != 0.0:
                    spec = spec * float(s["gain"]) + float(s["offset"])

                self.col_queue.append(spec)
                self.t_queue.append(dt.datetime.utcnow())
        except Exception:
            pass
        finally:
            self.stop_flag.set()

    def _invalidate_nf_proc(self):  # clears noise floor cache
        self.nf_proc = None
        self._nf_params = None

    def _prepare_nf(self, spec_len):  # if no noise floor loaded (yet) clear related vars
        if self.nf_raw is None:
            self.nf_proc = None
            self._nf_params = None
            return

        params = (spec_len,)
        if self._nf_params == params and self.nf_proc is not None:
            return

        nf = self.nf_raw.astype(np.float32)
        if nf.shape[0] != self.naxis1:
            nf_freq = np.linspace(self.freq.min(), self.freq.max(), nf.shape[0], endpoint=True, dtype=np.float32)
            nf = np.interp(self.freq, nf_freq, nf).astype(np.float32)  # 1d interp nf over the linspace defined by min/max freq

        self.nf_proc = nf
        self._nf_params = params

    #Image building
    def build_image(self):
        if self.reset_flag.is_set():
            self.wf = None
            self.tvec = []
            self.reset_flag.clear()

        updated = False
        while self.col_queue:
            col = self.col_queue.popleft()  # let out the data, left scrolling
            t   = self.t_queue.popleft()
            if self.wf is None:
                self.wf = col[:, None]  # establish if blank
                self.tvec = [t]
            else:
                if col.shape[0] != self.wf.shape[0]:  # do some scaling to stretch to window size until 2 minutes
                    self.wf = col[:, None]
                    self.tvec = [t]
                else:
                    if self.wf.shape[1] >= 1200:  # roll out and no window scale after window size filled, 1200 columns (2 min x 10Hz) in window
                        self.wf = np.hstack([self.wf[:, 1:], col[:, None]])
                        self.tvec = self.tvec[1:] + [t]
                    else:
                        self.wf = np.hstack([self.wf, col[:, None]])
                        self.tvec.append(t)
            updated = True

        if self.wf is None:  # check window still exists
            return None, updated

        s = self.state.snapshot()
        img = self.wf

        if s["use_db"]:  # dB mapping (scaled 16-bit log) Dr. Gray's method
            imgf = img.astype(np.float32, copy=True)
            m = imgf > 0.0
            imgf[m] = log_scale * np.log10(imgf[m])  # log scale = (65536 / (log65536) )
            imgf[~m] = 0.0  # anything less than 0 (not m) is 0
            img = imgf

        tx = self.tvec

        return (img, tx, s), updated

    def ensure_colorbar(self):  # map ye olde color bar to current vlims
        if self.im is None: return
        if self.cbar is None:
            sm = matplotlib.cm.ScalarMappable(norm=self.im.norm, cmap=self.im.get_cmap())
            self.cbar = self.fig.colorbar(sm, cax=self.cax)
        else:
            self.cbar.update_normal(matplotlib.cm.ScalarMappable(norm=self.im.norm, cmap=self.im.get_cmap()))

    def redraw(self, force=False):  # redraw the plot with new updates, allows scroll
        now = time.time()
        if not force and (now - self.last_draw < 0.03): #small redraw delay, ~33fps redraw
            return
        res, _ = self.build_image()
        if res is None: return
        img, tx, s = res
        if len(tx) < 2:
            return

        vmin, vmax = (float(s["vmin"]), float(s["vmax"]))
        if s["use_db"]:
            vmax = min(vmax, 65536.0)   # scaled-log full-scale cap? Or should I make the cap larger to allow for compressing display
            if vmax <= vmin: vmax = vmin + 1e-3
        else:
            if vmax <= vmin: vmax = vmin + 1.0

        cmap_obj = get_cmap(s["cmap"])
        interp = "bilinear" if s["interp_on"] else "none"
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

        y0, y1 = float(self.freq[0]), float(self.freq[-1])

        if self.im is None:  # Establish plot if blank
            self.im = self.ax.imshow(
                img, origin="lower", aspect="auto",
                interpolation=interp, cmap=cmap_obj,
                norm=norm,
                extent=[mdates.date2num(tx[0]), mdates.date2num(tx[-1]), y0, y1],
            )
            self.ensure_colorbar()
        else:  # fill appropriately if not blank
            self.im.set_data(img)
            self.im.set_extent([mdates.date2num(tx[0]), mdates.date2num(tx[-1]), y0, y1])
            self.im.set_cmap(cmap_obj)
            self.im.set_interpolation(interp)
            self.im.set_norm(norm)
            self.ensure_colorbar()

        label = "Intensity [dB 16b-Scaled]" if s["use_db"] else "Intensity" #cbar label
        self.cax.set_ylabel(label, rotation=270, labelpad=12)

        self.canvas.draw_idle()
        self.last_draw = now

    def on_timer(self):
        self.redraw(False)
        self.root.after(30, self.on_timer)

def main():
    app = App()
    app.root.mainloop()

if __name__ == "__main__":
    main()
