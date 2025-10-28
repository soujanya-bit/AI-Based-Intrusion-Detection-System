import numpy as np
import pandas as pd
from collections import defaultdict

def extract_flow_features(packets):
    """
    Extracts statistical features from a list of Scapy packets, correctly handling
    bidirectional flows.
    """
    # The key is a canonical representation of the flow (sorted IPs and ports)
    # The value is a dictionary holding the packets and the forward direction
    flows = defaultdict(lambda: {'packets': [], 'forward_direction': None})

    for pkt in packets:
        if not pkt.haslayer('IP'):
            continue

        src_ip = pkt['IP'].src
        dst_ip = pkt['IP'].dst
        proto = pkt['IP'].proto
        src_port = pkt['TCP'].sport if pkt.haslayer('TCP') else pkt['UDP'].sport if pkt.haslayer('UDP') else 0
        dst_port = pkt['TCP'].dport if pkt.haslayer('TCP') else pkt['UDP'].dport if pkt.haslayer('UDP') else 0

        # Create a canonical flow key by sorting IPs and ports.
        # This groups (A->B) and (B->A) into the same flow.
        ip_ports = sorted([(src_ip, src_port), (dst_ip, dst_port)])
        flow_key = (ip_ports[0][0], ip_ports[1][0], ip_ports[0][1], ip_ports[1][1], proto)

        # The first packet of the flow defines the forward direction
        if not flows[flow_key]['forward_direction']:
            flows[flow_key]['forward_direction'] = (src_ip, src_port, dst_ip, dst_port)

        # Determine if the packet is forward or backward
        direction = 'fwd' if (src_ip, src_port, dst_ip, dst_port) == flows[flow_key]['forward_direction'] else 'bwd'
        
        timestamp = float(pkt.time)
        pkt_len = len(pkt)
        flows[flow_key]['packets'].append({'timestamp': timestamp, 'len': pkt_len, 'dir': direction})

    flow_features = []
    for flow_key, flow_data in flows.items():
        packets_in_flow = flow_data['packets']
        if len(packets_in_flow) < 2:
            continue

        # Ensure packets are sorted by time before calculating features
        packets_in_flow.sort(key=lambda p: p['timestamp'])

        timestamps = np.array([p['timestamp'] for p in packets_in_flow])
        sizes = np.array([p['len'] for p in packets_in_flow])
        
        fwd_packets = [p for p in packets_in_flow if p['dir'] == 'fwd']
        bwd_packets = [p for p in packets_in_flow if p['dir'] == 'bwd']

        total_fwd_packets = len(fwd_packets)
        total_bwd_packets = len(bwd_packets)
        
        flow_duration = timestamps[-1] - timestamps[0]

        fwd_pkt_lengths = np.array([p['len'] for p in fwd_packets])
        bwd_pkt_lengths = np.array([p['len'] for p in bwd_packets])

        fwd_timestamps = np.array([p['timestamp'] for p in fwd_packets])
        bwd_timestamps = np.array([p['timestamp'] for p in bwd_packets])

        fwd_iats = np.diff(fwd_timestamps) if total_fwd_packets > 1 else np.array([])
        bwd_iats = np.diff(bwd_timestamps) if total_bwd_packets > 1 else np.array([])

        features = {
            'Flow Duration': flow_duration,
            'Total Fwd Packets': total_fwd_packets,
            'Total Backward Packets': total_bwd_packets,
            'Fwd Packet Length Mean': np.mean(fwd_pkt_lengths) if total_fwd_packets > 0 else 0,
            'Bwd Packet Length Mean': np.mean(bwd_pkt_lengths) if total_bwd_packets > 0 else 0,
            'Flow Bytes/s': np.sum(sizes) / flow_duration if flow_duration > 0 else 0,
            'Flow Packets/s': len(packets_in_flow) / flow_duration if flow_duration > 0 else 0,
            'Fwd IAT Mean': np.mean(fwd_iats) if len(fwd_iats) > 0 else 0,
            'Bwd IAT Mean': np.mean(bwd_iats) if len(bwd_iats) > 0 else 0,
            'Packet Length Mean': np.mean(sizes),
        }
        flow_features.append(features)

    return pd.DataFrame(flow_features) if flow_features else pd.DataFrame()
