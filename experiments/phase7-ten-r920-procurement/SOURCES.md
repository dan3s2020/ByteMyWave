# Sources and procurement-status map

Date checked: **2026-08-17**

This file distinguishes official hardware documentation from market listings. Market listings are volatile and must be re-checked before purchase.

## Official / manufacturer references

### Dell PowerEdge R920

- R920 owner's/technical documentation: https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/
- Expansion-card guidelines: https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/expansion-card-installation-guidelines

Project facts retained from Dell documentation:

```text
96 DIMM sockets
DDR3 ECC server memory
4-socket platform
with four CPUs, six native x16 links used in project topology at slots 4-9
```

Mechanical fit and power support for six consumer GPUs is **not** claimed by Dell; the project uses external frames and true x16 extensions.

### Intel Xeon E7-4890 v2

- Intel specifications: https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html

Retained facts:

```text
15 cores / 30 threads
2.8GHz base
155W TDP
4 memory channels/socket
85GB/s published max memory bandwidth
AVX, no AVX2/AVX-512/AMX
```

### NVIDIA RTX 3060

- NVIDIA RTX 3060 family: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/

Project target is specifically the **RTX 3060 12GB**.

### NVIDIA GTX 1060

- NVIDIA GTX 1060 announcement/spec context: https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/

Project target is specifically **GTX 1060 6GB desktop-class GP106**, used as a custom-kernel expert worker.

### Comino x16 riser

- https://www.comino.com/products/comino-x16-x16-universal-riser

Required because project performance relies on true x16 connectivity and separate slot power. Price is RFQ; do not substitute x1 mining risers.

## Current/volatile procurement listings

### R920 liquidation target

- Alibaba barebones example: https://www.alibaba.com/pla/Used-Second-Hand-Barebones-PowerEdge-R920_1601229763698.html

Status: **reference only / not approved for bulk order**. Listing is barebones and may lack required memory risers/CPU/PSU parts. Supplier must quote a complete configuration and provide internal photos.

### E7-4890 v2

- Piospartslap: https://www.piospartslap.de/Intel-Xeon-Processor-E7-4890-V2-15-Core-375MB-Cache-280-GHz-FCLGA-2011-SR1GL

Status: fallback if R920 lot lacks CPUs.

### DDR3 ECC RDIMM

- OLX: https://www.olx.ro/d/oferta/memorii-ram-server-ddr3-ecc-reg-samsung-hynix-micron-nanya-8gb-16gb-hp-IDkeXN4.html

Status: current local price reference at ~80 lei/16GB. Need confirmation of 160-piece quantity and exact part/rank.

### RTX 3060 12GB

- OLX current-search pool: https://www.olx.ro/electronice-si-electrocasnice/componente-laptop-pc/placi-video/q-placa-video-rtx-3060-12gb/

Status: multiple individual sellers rather than one guaranteed ten-card lot. Test every used board.

### GTX 1060 6GB bulk

- Alibaba: https://www.alibaba.com/product-detail/In-Stock-GTX-1060-6GB-Gaming_1600887743749.html

Status: bulk target; verify exact 6GB desktop card, GPU identity, photos, benchmark/test policy, DOA policy, shipping and tax before payment.

### External PSU candidate

- Cougar Polar 1200W at ForIT: https://www.forit.ro/surse-pc/cougar/416155-polar-80-plus-platinum-1200w/

Status: candidate, not frozen until exact GPU auxiliary connector count and riser-power method are validated.

### FDR NIC

- MCX353A-class listing: https://www.ebay.com/itm/358431102071
- Full-height bracket fallback: https://www.ebay.co.uk/itm/286670417723

Status: x8 card deliberately chosen so all six R920 x16 links remain available to GPUs.

### FDR switch

- SX6036: https://insidesystems.com/product/sx6036/

Status: 36-port FDR switch class; confirm firmware, fans, PSU and management access with seller.

### FDR DAC

- Mellanox-compatible cable: https://www.ebay.com/itm/162826133336

Status: verify cable length against final rack placement.

### Storage/caddy

- SSD: https://www.okazii.ro/solid-state-drive-ssd-240gb-sata-6-0gb-s-diferite-modele-a229936595
- Dell caddy candidate: https://www.hddcaddy.ro/SAS-SATA-Dell-PowerEdge-G176J

### Rack/rails/frame

- R920 rails: https://www.ebay.de/itm/397674080158
- 42U open rack: https://qmart.ro/cabinete-metalice-rack-deschis-19-42u-600x1000-negru-or01-6042-b
- external 6-GPU frame: https://www.olx.ro/d/oferta/carcasa-rig-ai-mining-cadru-open-air-rack-minare-placi-video-etc-rvn-eth-btc-machinelearning-IDkd4KS.html

A 42U rack has no spare rack units with ten 4U R920s plus two 1U switches. Prefer 45/47U if a similar-priced unit is available.

### Power distribution candidate

- APC AP7553: https://www.senetic.ro/product/AP7553

Status: only a rack-distribution candidate. Building circuits, phases, breakers, grounding and HVAC remain site-specific professional work.

### Management network

- TP-Link TL-SG1024D: https://www.okazii.ro/switch-tp-link-tl-sg1024d-cu-24-porturi-rj45-gigabit-unmanaged-1u-a263436362
- Cat6: https://www.senetic.ro/product/733229

## Seller questions that must be answered before payment

### R920 seller

```text
Is each server complete with all 8 memory riser boards?
Are four CPU positions populated/usable?
Which CPUs are installed?
Are all CPU heatsinks and fan modules present?
Which PSU modules and how many per server?
Are all PCIe expansion/riser boards and slot hardware present?
Does iDRAC boot without critical errors?
Can you provide internal photos of every server or representative units from the same lot?
Can you quote pallet freight to Romania?
What is the DOA/return policy?
```

### GTX1060 bulk seller

```text
Confirm 50 x GTX 1060 6GB desktop cards.
Confirm GP106 and 6GB GDDR5; no 3GB/5GB/P106 substitution.
Provide photos of actual lot and model mix.
Confirm PCIe enumeration/test result for every unit.
Provide DOA rate/return/replacement policy.
Quote DDP/DAP shipping to Romania and package dimensions/weight.
```

### RAM seller

```text
Need 160 x 16GB DDR3-1600 ECC Registered modules.
Provide exact manufacturer part numbers, ranks and voltage.
Confirm whether all 160 can be same part number or split into matched groups.
Confirm tested status and replacement policy.
```

### Comino / powered riser supplier

```text
Need first 6 units, then up to 60 total.
Confirm full PCIe Gen3 x16 lane support at chosen cable length.
Confirm separate slot-power specification and connector type.
Quote 6-unit prototype and 60-unit bulk price to Romania.
Provide recommended power wiring limits and installation documentation.
```

## Source hierarchy

For technical claims:

```text
manufacturer docs > official model/config/source > measured project benchmark > market listing
```

Market listings establish procurement possibilities and prices only. They are not used as proof of electrical bandwidth, kernel performance or K3 tok/s.