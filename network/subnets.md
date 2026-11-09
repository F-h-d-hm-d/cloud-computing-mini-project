# Network Subnet Design

**Student:** Sameer Hidayat
**Roll Number:** 230101064

## VPC Network
Base CIDR: 10.20.0.0/16

## VLSM Subnet Allocation

| Subnet | CIDR | Usable Hosts | Purpose |
|---|---|---:|---|
| Application | 10.20.0.0/22 | 1022 | Flask inference application |
| Public | 10.20.4.0/24 | 254 | Ingress and external routing |
| Data | 10.20.5.0/25 | 126 | SeaweedFS and datasets |
| Management | 10.20.5.128/27 | 30 | Cluster administration |

## Network Security

I implemented inbound and outbound Network Access Control List (NACL) rules in network/nacl.json.

The rules include a deny entry for 198.51.100.0/24.

## Validation

I tested the network configuration using tools/nacl_sim.py.

Results:
- Packet simulation: 6/6 passed
- Network unit tests: 3/3 passed
- Subnet allocations: Non-overlapping

## Implementation Note

This subnet plan represents the project's simulated cloud network design. The NACL rules were validated using a Python simulator rather than applied to a public-cloud VPC.
