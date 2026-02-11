# FPGA Reservoir Computing Setup Guide  
**PYNQ-Z1 + Vivado + hls4ml**

This guide explains how to:

- Set up the **PYNQ-Z1 board**
- Install **Vivado (2019.2 / 2020.1)**
- Use **hls4ml** to generate FPGA bitstreams
- Deploy models to the PYNQ board

📄 **Full detailed guide (PDF version):**  
Attached in this very repository, with the name FPGA_RC_Setup_Guide.pdf

---

This README provides a structured, GitHub-friendly version of the guide.

---

# 📌 System Requirements

- Linux (x86-64 architecture required)
- Tested on Debian (kernel 6.x)
- PYNQ-Z1 board
- Ethernet cable + USB cable
- ~30GB free disk space on your PC(Vivado is large)

> ⚠️ This guide assumes a Linux environment. If you use Windows or macOS, consult alternative installation guides.

---

# 1️⃣ PYNQ-Z1 Setup

## 1.1 Connect the Board to your PC

1. Connect:
   - Ethernet cable
   - USB cable
2. Power on the board.
3. Wait until:
   - Blue LEDs (LD4–LD5) blink
   - Yellow/Green LEDs (LD0–LD3) stay on

Board is now booted.

Official reference:
- https://pynq.readthedocs.io/en/v2.6.1/getting_started/pynq_z1_setup.html

---

## 1.2 Configure Ethernet (Static IP) on your PC

Check available interfaces:

```bash
ip link
```

Assign static IP (NetworkManager systems):

```bash
nmcli con show
nmcli con mod "interface_name" ipv4.method manual ipv4.addresses 192.168.2.1/24
nmcli con up "interface_name"
```

Restart networking:

```bash
sudo systemctl restart networking
```

Ping the board:

```bash
ping 192.168.2.99
```

Access Jupyter:

```
http://192.168.2.99:9090
```

Credentials:

```
username: xilinx
password: xilinx
```

To open terminal in Jupyter:
```
New → Terminal
```

---

## 1.3 Provide Internet to the PYNQ Board (NAT Setup on your PC)

### Enable IP forwarding

Edit `/etc/sysctl.conf`:

```
net.ipv4.ip_forward=1
```

---

### Configure NAT (example)

Assumptions:
- Internet interface: `wlo1`
- PYNQ interface: `eth0`
- PYNQ network: `192.168.2.0/24`

```bash
sudo iptables -t nat -A POSTROUTING -o wlo1 -s 192.168.2.0/24 -j MASQUERADE
sudo iptables -I FORWARD 1 -i eth0 -o wlo1 -j ACCEPT
sudo iptables -I FORWARD 2 -i wlo1 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
```

Check rules:

```bash
sudo iptables -t nat -L -n -v
sudo iptables -L FORWARD -v -n
```

Make persistent:

```bash
sudo apt install iptables-persistent
sudo netfilter-persistent save
```

---

### Configure PYNQ network(on the board)

Edit:

```
/etc/network/interfaces.d/eth0
```

Add:

```
auto eth0
iface eth0 inet static
address 192.168.2.99
netmask 255.255.255.0
gateway 192.168.2.1
dns-nameservers 8.8.8.8
```

Restart:

```bash
sudo systemctl restart networking
```

Test:

```bash
ping 192.168.2.1
ping 8.8.8.8
ping google.com
```

---

# 2️⃣ Install Vivado(on your PC)

## Recommended Versions

- Preferred: **Vivado HLS 2020.1**
- If errors occur: **Vivado 2019.2** (tested working on Debian - kernel 6.x)

Download from:
https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vivado-design-tools/archive.html

---

## Installation Steps

Extract:

```bash
tar -xzf Xilinx_Vivado_2019.2_1106_2127.tar.gz
cd Xilinx_Vivado_2019.2_1106_2127
sudo ./xsetup
```

Ignore suggestion to install newer version.

Select minimal installation options.

---

## Add Vivado to PATH

Edit:

```bash
nano ~/.bashrc
```

Add:

```bash
source /tools/Xilinx/Vivado/2019.2/settings64.sh
```

Reload:

```bash
source ~/.bashrc
```

Launch:

```bash
vivado
```

---

### Fix Common Error

If you see:

```
librdi_commontasks.so: libtinfo.so.5 not found
```

Install:

```bash
sudo apt-get install libtinfo5 libncurses5
```

---

# 3️⃣ Using hls4ml

## 3.1 Install (PC Side)

Install on your PC (NOT the PYNQ board):

```bash
pip install hls4ml
```

Tutorial repository to make a simple Neural Network:

https://github.com/fastmachinelearning/hls4ml-tutorial

---

## Minimum Required Tutorials(The rest are for optimization)

- `part1_getting_started`
- `part7a_bitstream`
- `part7b_deployment`

Bitstream creation line:

```python
hls_model.build(csim=False, export=True, bitfile=True)
```

This step takes time... Comment it if you just want to do inference on your PC, and don't want to create a bitstream for the FPGA.

---

## Modifications we made to the official guide:

- Converted from Keras → PyTorch
- Removed `softmax` (heavy for FPGA build)
- Build optimizations marked as:

```python
# Modified to optimize
```

You can find the modified files in this repository with the names:

- `part1_getting_started.py`
- `part7_bitstream.py`

---

## 3.2 On the PYNQ Board

If deployment errors occur:

File likely responsible:

```
axi_stream_driver.py
```

Fix:
- Replace `allocate` module
- Use `Xlnk` module instead

This file handles data I/O between bitstream and system.

---

# ⚠️ Common Pitfalls

- Vivado version mismatch
- Missing system libraries
- Driver issues in `axi_stream_driver.py`
