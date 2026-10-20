#!/usr/bin/env python3
"""
Fixed full extraction for any MAWI PCAP - with proper error handling.
Usage: python extract_fixed.py <date>  (e.g., 20210210, 20211013)
"""

import gzip
import dpkt
import socket
import time
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
from collections import defaultdict

if len(sys.argv) < 2:
    print("Usage: python extract_fixed.py <YYYYMMDD>")
    sys.exit(1)

DATE = sys.argv[1]
PCAP_PATH = f'mawi_2021/pcap/{DATE}1400.pcap.gz'
OUTPUT_DIR = f'mawi_2021/flows_{DATE}_full'
CHUNK_SIZE = 50000
IDLE_TIMEOUT = 120.0
HARD_TIMEOUT = 300.0
MAX_ACTIVE = 500000

if not os.path.exists(PCAP_PATH):
    print(f"PCAP not found: {PCAP_PATH}")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class FlowState:
    flow_key: Tuple
    start_ts: float
    last_ts: float
    first_pkt_dir: bool
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_pkt_lens: List[int] = field(default_factory=list)
    bwd_pkt_lens: List[int] = field(default_factory=list)
    fwd_iat: List[float] = field(default_factory=list)
    bwd_iat: List[float] = field(default_factory=list)
    last_fwd_ts: Optional[float] = None
    last_bwd_ts: Optional[float] = None
    fwd_flags: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    bwd_flags: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fwd_fin_seen: bool = False
    bwd_fin_seen: bool = False
    fwd_rst_seen: bool = False
    bwd_rst_seen: bool = False
    
    def add_packet(self, ts, ip_len, is_fwd, tcp_flags=0, proto=0):
        self.last_ts = ts
        if is_fwd:
            self.fwd_pkts += 1; self.fwd_bytes += ip_len; self.fwd_pkt_lens.append(ip_len)
            if self.last_fwd_ts is not None: self.fwd_iat.append(ts - self.last_fwd_ts)
            self.last_fwd_ts = ts
            if proto == 6:
                self._flags(tcp_flags, self.fwd_flags)
                if tcp_flags & dpkt.tcp.TH_FIN: self.fwd_fin_seen = True
                if tcp_flags & dpkt.tcp.TH_RST: self.fwd_rst_seen = True
        else:
            self.bwd_pkts += 1; self.bwd_bytes += ip_len; self.bwd_pkt_lens.append(ip_len)
            if self.last_bwd_ts is not None: self.bwd_iat.append(ts - self.last_bwd_ts)
            self.last_bwd_ts = ts
            if proto == 6:
                self._flags(tcp_flags, self.bwd_flags)
                if tcp_flags & dpkt.tcp.TH_FIN: self.bwd_fin_seen = True
                if tcp_flags & dpkt.tcp.TH_RST: self.bwd_rst_seen = True
    
    def _flags(self, flags, counter):
        for bit, name in [(dpkt.tcp.TH_FIN,'fin'),(dpkt.tcp.TH_SYN,'syn'),(dpkt.tcp.TH_RST,'rst'),
                          (dpkt.tcp.TH_PUSH,'psh'),(dpkt.tcp.TH_ACK,'ack'),(dpkt.tcp.TH_URG,'urg'),
                          (dpkt.tcp.TH_ECE,'ece'),(dpkt.tcp.TH_CWR,'cwr')]:
            if flags & bit: counter[name] += 1
    
    def is_closed(self):
        return (self.fwd_fin_seen or self.fwd_rst_seen) and (self.bwd_fin_seen or self.bwd_rst_seen)
    
    def to_dict(self):
        tp = self.fwd_pkts + self.bwd_pkts
        tb = self.fwd_bytes + self.bwd_bytes
        dur = self.last_ts - self.start_ts
        (ip1,p1),(ip2,p2),proto = self.flow_key
        if self.first_pkt_dir:
            sip,sport,dip,dport = ip1,p1,ip2,p2
        else:
            sip,sport,dip,dport = ip2,p2,ip1,p1
        def st(a):
            if not a: return (0.0,0.0,0.0,0.0)
            a = np.array(a, dtype=np.float32)
            return (float(a.min()), float(a.max()), float(a.mean()), float(a.std()))
        fl = st(self.fwd_pkt_lens); bl = st(self.bwd_pkt_lens)
        al = st(self.fwd_pkt_lens + self.bwd_pkt_lens)
        fi = st(self.fwd_iat); bi = st(self.bwd_iat)
        ai = st(self.fwd_iat + self.bwd_iat)
        return {
            'flow_id': hash(self.flow_key) & 0xFFFFFFFFFFFFFFFF,
            'src_ip': sip, 'dst_ip': dip, 'src_port': sport, 'dst_port': dport, 'protocol': proto,
            'start_ts': self.start_ts, 'end_ts': self.last_ts, 'duration': dur,
            'fwd_pkts': self.fwd_pkts, 'bwd_pkts': self.bwd_pkts, 'total_pkts': tp,
            'fwd_bytes': self.fwd_bytes, 'bwd_bytes': self.bwd_bytes, 'total_bytes': tb,
            'fwd_pkt_len_min': fl[0], 'fwd_pkt_len_max': fl[1], 'fwd_pkt_len_mean': fl[2], 'fwd_pkt_len_std': fl[3],
            'bwd_pkt_len_min': bl[0], 'bwd_pkt_len_max': bl[1], 'bwd_pkt_len_mean': bl[2], 'bwd_pkt_len_std': bl[3],
            'pkt_len_min': al[0], 'pkt_len_max': al[1], 'pkt_len_mean': al[2], 'pkt_len_std': al[3],
            'fwd_iat_min': fi[0], 'fwd_iat_max': fi[1], 'fwd_iat_mean': fi[2], 'fwd_iat_std': fi[3],
            'bwd_iat_min': bi[0], 'bwd_iat_max': bi[1], 'bwd_iat_mean': bi[2], 'bwd_iat_std': bi[3],
            'iat_min': ai[0], 'iat_max': ai[1], 'iat_mean': ai[2], 'iat_std': ai[3],
            'fwd_syn_cnt': self.fwd_flags.get('syn',0), 'fwd_fin_cnt': self.fwd_flags.get('fin',0),
            'fwd_rst_cnt': self.fwd_flags.get('rst',0), 'fwd_psh_cnt': self.fwd_flags.get('psh',0),
            'fwd_ack_cnt': self.fwd_flags.get('ack',0), 'fwd_urg_cnt': self.fwd_flags.get('urg',0),
            'fwd_ece_cnt': self.fwd_flags.get('ece',0), 'fwd_cwr_cnt': self.fwd_flags.get('cwr',0),
            'bwd_syn_cnt': self.bwd_flags.get('syn',0), 'bwd_fin_cnt': self.bwd_flags.get('fin',0),
            'bwd_rst_cnt': self.bwd_flags.get('rst',0), 'bwd_psh_cnt': self.bwd_flags.get('psh',0),
            'bwd_ack_cnt': self.bwd_flags.get('ack',0), 'bwd_urg_cnt': self.bwd_flags.get('urg',0),
            'bwd_ece_cnt': self.bwd_flags.get('ece',0), 'bwd_cwr_cnt': self.bwd_flags.get('cwr',0),
            'fwd_pkts_ratio': self.fwd_pkts/tp if tp>0 else 0.0,
            'fwd_bytes_ratio': self.fwd_bytes/tb if tb>0 else 0.0,
        }

def make_key(sip, dip, sp, dp, pr):
    e1, e2 = (sip, sp), (dip, dp)
    return (e1, e2, pr) if e1 <= e2 else (e2, e1, pr)

class Extractor:
    def __init__(self):
        self.flows = {}
        self.buf = []
        self.chunk_idx = 0
        self.total_pkts = 0
        self.total_flows = 0
        self.t0 = time.time()
        self.last_report = 0
        self.last_pkts = 0
        existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith('flows_chunk_') and f.endswith('.parquet')])
        if existing:
            self.chunk_idx = int(existing[-1].split('_')[-1].split('.')[0]) + 1
            print(f"Resuming {DATE} from chunk {self.chunk_idx} ({len(existing)} existing)")

    def emit(self, key):
        fl = self.flows.pop(key)
        self.buf.append(fl.to_dict())
        self.total_flows += 1
        if len(self.buf) >= CHUNK_SIZE:
            self.flush()

    def flush(self):
        if not self.buf: return
        df = pd.DataFrame(self.buf)
        for c in df.columns:
            if df[c].dtype == 'float64': df[c] = df[c].astype('float32')
            elif df[c].dtype == 'int64': df[c] = df[c].astype('int32')
        out = os.path.join(OUTPUT_DIR, f'flows_chunk_{self.chunk_idx:04d}.parquet')
        df.to_parquet(out, compression='snappy', index=False)
        sz = os.path.getsize(out) / 1024**2
        print(f"  [{DATE}] Chunk {self.chunk_idx:04d}: {len(df):,} flows -> {sz:.1f} MB")
        self.buf.clear()
        self.chunk_idx += 1

    def expire(self, ts):
        to_exp = []
        for k, fl in self.flows.items():
            if fl.is_closed() or ts - fl.last_ts > IDLE_TIMEOUT or ts - fl.start_ts > HARD_TIMEOUT:
                to_exp.append(k)
        if len(self.flows) - len(to_exp) > MAX_ACTIVE:
            inactive = [(k,f) for k,f in self.flows.items() if k not in to_exp]
            inactive.sort(key=lambda x: x[1].last_ts)
            for k,_ in inactive[:len(inactive) - (MAX_ACTIVE - len(to_exp))]:
                to_exp.append(k)
        for k in to_exp: self.emit(k)

    def run(self):
        print(f"=== {DATE} EXTRACTION START ===")
        pkts_since_report = 0
        with gzip.open(PCAP_PATH, 'rb') as f:
            reader = dpkt.pcap.Reader(f)
            for ts, buf in reader:
                self.total_pkts += 1
                pkts_since_report += 1
                
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if isinstance(eth.data, dpkt.ip.IP):
                        ip = eth.data
                        sip = socket.inet_ntoa(ip.src); dip = socket.inet_ntoa(ip.dst)
                        pr = ip.p; ip_len = ip.len
                        sp = dp = 0; tf = 0
                        if pr == 6 and isinstance(ip.data, dpkt.tcp.TCP):
                            tcp = ip.data; sp, dp = tcp.sport, tcp.dport; tf = tcp.flags
                        elif pr == 17 and isinstance(ip.data, dpkt.udp.UDP):
                            udp = ip.data; sp, dp = udp.sport, udp.dport
                        elif pr == 1:
                            icmp = ip.data; sp, dp = icmp.type, icmp.code
                    elif isinstance(eth.data, dpkt.ip6.IP6):
                        ip = eth.data
                        sip = socket.inet_ntop(socket.AF_INET6, ip.src)
                        dip = socket.inet_ntop(socket.AF_INET6, ip.dst)
                        pr = ip.nxt; ip_len = ip.plen + 40
                        sp = dp = 0; tf = 0
                    else:
                        continue
                    
                    key = make_key(sip, dip, sp, dp, pr)
                    (a1,p1),(a2,p2),_ = key
                    fwd = (sip == a1 and dip == a2 and sp == p1 and dp == p2)
                    
                    if key not in self.flows:
                        self.flows[key] = FlowState(key, ts, ts, fwd)
                    self.flows[key].add_packet(ts, ip_len, fwd, tf, pr)
                    
                except Exception as e:
                    if self.total_pkts % 1000000 == 0:
                        print(f"  [{DATE}] Packet parse error: {e}")
                    continue

                if self.total_pkts % 100000 == 0:
                    self.expire(ts)
                
                if self.total_pkts % 1000000 == 0:
                    et = time.time() - self.t0
                    rate = self.total_pkts / et
                    inst = (self.total_pkts - self.last_pkts) / (et - self.last_report) if et != self.last_report else 0
                    print(f"  [{DATE}] PKTS: {self.total_pkts:>12,} | ACTIVE: {len(self.flows):>7,} | EMITTED: {self.total_flows:>8,} | AVG: {rate:>8.0f} p/s | INST: {inst:>8.0f} p/s | TIME: {et:>7.1f}s")
                    self.last_report = et
                    self.last_pkts = self.total_pkts

        print(f"[{DATE}] Expiring remaining flows ({len(self.flows)} active)...")
        self.expire(ts + HARD_TIMEOUT + 1)
        self.flush()
        et = time.time() - self.t0
        sz = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if f.endswith('.parquet'))
        print(f"=== {DATE} DONE === PKTS: {self.total_pkts:,} FLOWS: {self.total_flows:,} SIZE: {sz/1024**2:.1f}MB TIME: {et:.1f}s")

if __name__ == '__main__':
    Extractor().run()