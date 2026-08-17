# Procurement BOM — 10x R920 Kimi K3 cluster

Date checked: **2026-08-17**

This BOM records the currently verified purchase targets and separates **known public prices** from **conversation target prices** and **RFQ-only items**. Availability and shipping can change; re-check before purchase.

## 1. Compute/server hardware

| Item | Qty | Planning/current price | Extended | Purchase/source | Notes |
|---|---:|---:|---:|---|---|
| Dell PowerEdge R920 | 10 | **1,000 lei target/node** | 10,000 lei target | https://www.alibaba.com/pla/Used-Second-Hand-Barebones-PowerEdge-R920_1601229763698.html | The linked Alibaba unit is advertised as **barebones** and mentions a single memory board; it is **not automatically acceptable**. Supplier must quote complete nodes with all memory risers, four CPU positions usable, heatsinks/fans, PCIe hardware and PSUs. The 1,000 lei number is a liquidation target, not a reproducible refurb-market price. |
| Xeon E7-4890 v2 | up to 40 | ~EUR19.90 each observed | ~4.2k lei if needed | https://www.piospartslap.de/Intel-Xeon-Processor-E7-4890-V2-15-Core-375MB-Cache-280-GHz-FCLGA-2011-SR1GL | Buy only if target R920 lot does not already include CPUs. |
| 16GB DDR3-1600 ECC Registered | 160 | **80 lei each planning/current local listing** | **12,800 lei** | https://www.olx.ro/d/oferta/memorii-ram-server-ddr3-ecc-reg-samsung-hynix-micron-nanya-8gb-16gb-hp-IDkeXN4.html | Need supplier confirmation for 160 pieces and exact part/rank. Do not mix RDIMM and LRDIMM. 16 DIMMs/node = one DIMM per native channel across four CPUs. |
| RTX 3060 12GB | 10 | ~900-1,050 lei observed | ~9,000-10,500 lei | https://www.olx.ro/electronice-si-electrocasnice/componente-laptop-pc/placi-video/q-placa-video-rtx-3060-12gb/ | Must be the **12GB** model. Prefer similar board layouts/power connectors for serviceability. |
| GTX 1060 6GB worker | 50 | ~$102-115 observed bulk listing | ~23.1k-26.0k lei before freight/tax | https://www.alibaba.com/product-detail/In-Stock-GTX-1060-6GB-Gaming_1600887743749.html | Require real desktop GTX1060 6GB/GP106, PCIe x16. Reject 3GB, 5GB, P106 mining-only substitutions unless separately benchmarked. |

## 2. PCIe and external GPU mounting

| Item | Qty | Price | Source | Acceptance condition |
|---|---:|---:|---|---|
| **Powered true x16->x16 PCIe riser** | 60 | **RFQ** | https://www.comino.com/products/comino-x16-x16-universal-riser | Must carry all x16 lanes and provide separate slot power. Do **not** substitute x1 mining risers. Buy six first and benchmark before ordering remaining 54. |
| External six-GPU open frame | 10 | ~325 lei/node planning | https://www.olx.ro/d/oferta/carcasa-rig-ai-mining-cadru-open-air-rack-minare-placi-video-etc-rvn-eth-btc-machinelearning-IDkd4KS.html | Needs room for 1x3060 + 5x1060 and direct airflow. |
| 1200W-class external ATX PSU | 10 | ~993 lei observed candidate | https://www.forit.ro/surse-pc/cougar/416155-polar-80-plus-platinum-1200w/ | Candidate: Cougar Polar 1200W Platinum. Must provide enough native PCIe auxiliary outputs plus riser-power capability. Verify exact GPU connector mix before purchase. |

### Why powered x16 risers are mandatory

The performance model assumes approximately 10-12GB/s sustained host-to-device traffic over each PCIe Gen3 x16 worker feed. A common mining x1->x16 riser destroys this assumption. Likewise, letting six external GPUs draw their slot power through unsupported server-slot cabling is not accepted as the final electrical design.

## 3. Low-latency interconnect

| Item | Qty | Price | Source | Notes |
|---|---:|---:|---|---|
| Mellanox ConnectX-3 MCX353A-class FDR56, PCIe x8 | 10 | ~US$48 observed | https://www.ebay.com/itm/358431102071 | x8 is deliberate: keep the six x16 R920 links free for 1x3060 + 5 workers. Confirm full-height bracket. |
| Full-height NIC bracket if needed | 10 | ~US$9 | https://www.ebay.co.uk/itm/286670417723 | Only if NIC listing lacks proper bracket. |
| Mellanox SX6036 36-port FDR56 switch | 1 | ~EUR158 ex VAT observed | https://insidesystems.com/product/sx6036/ | 36 ports are enough for ten compute nodes plus expansion. |
| Mellanox-compatible FDR QSFP DAC ~1m | 10 | ~US$35 observed | https://www.ebay.com/itm/162826133336 | Rack placement must keep all compute nodes within cable reach. |
| 24-port 1GbE management switch | 1 | ~431 lei observed candidate | https://www.okazii.ro/switch-tp-link-tl-sg1024d-cu-24-porturi-rj45-gigabit-unmanaged-1u-a263436362 | Management/iDRAC network only, not model collectives. |
| Cat6 patch cables | ~20 | low-cost | https://www.senetic.ro/product/733229 | Management/iDRAC. |

InfiniBand requires a subnet manager (for example OpenSM) running somewhere in the fabric. Do not assume GPUDirect RDMA works on this exact consumer-GPU/old-server topology until measured. The runtime can still use host-pinned buffers and RDMA/verbs where supported.

## 4. Boot/storage, rack and rails

| Item | Qty | Price | Source | Notes |
|---|---:|---:|---|---|
| 240GB SATA SSD | 10 | ~179 lei | https://www.okazii.ro/solid-state-drive-ssd-240gb-sata-6-0gb-s-diferite-modele-a229936595 | OS/runtime/logs. Model weights live primarily in distributed RAM/storage workflow, not this boot drive. |
| Dell 2.5in caddy | 10 | ~42.47 lei | https://www.hddcaddy.ro/SAS-SATA-Dell-PowerEdge-G176J | Verify exact R920 bay/caddy compatibility before bulk order. |
| R920 ReadyRails | 10 | ~EUR50/node observed bulk tier | https://www.ebay.de/itm/397674080158 | Skip if R920 liquidation lot includes rails. |
| 42U open rack 600x1000 | 1 | ~1,123 lei | https://qmart.ro/cabinete-metalice-rack-deschis-19-42u-600x1000-negru-or01-6042-b | Ten R920s at 4U each plus two 1U switches exactly consumes 42U. Prefer 45/47U if available at similar cost. External GPU frames do not live inside this 42U calculation. |

## 5. Facility power distribution

| Item | Qty | Price | Source | Notes |
|---|---:|---:|---|---|
| APC AP7553 32A/230V rack PDU class | 3 | ~2,747 lei each | https://www.senetic.ro/product/AP7553 | Candidate distribution hardware. Final branch/circuit/PDU design must be approved by a qualified electrician based on actual measured load and facility supply. |

Do not infer building wiring from this BOM. The cluster planning envelope is roughly 16-20kW at the wall under sustained heavy load, and cooling adds facility demand.

## 6. Known subtotal

Using the currently discussed target prices and excluding optional CPUs where servers already include them:

```text
10 R920 target                     10,000 lei
160 x 16GB DDR3                    12,800
10 x RTX3060                       9,000-10,500
50 x GTX1060                       ~23,100-26,000 before freight/tax
10 x 1200W external PSU            ~9,930
10 x FDR NIC                       ~2,175
NIC brackets                       ~416 if needed
SX6036                             ~830 + VAT
10 x FDR DAC                       ~1,585
10 x SSD                           ~1,790
10 x caddy                         ~425
rails                              ~2,635 if not included
rack                               ~1,123
10 x GPU frame                     ~3,250
3 x PDU                            ~8,240
management switch                  ~431
management cabling                 ~145
---------------------------------------------------------
known planning subtotal            ~86k-92k lei class
```

Still excluded:

```text
60 x powered true-x16 risers (RFQ)
shipping/freight
VAT/customs where applicable
spares/replacements
HVAC / heat extraction
facility electrical work
labor/rework
```

Therefore `100,000 lei` is a **tight cap**, not a guaranteed installed-system total. `~120,000 lei` is a safer maximum project envelope while still purchasing in stages.

## 7. First-node purchase pack

Do not start with the x10 order. First procure:

```text
1 x verified-complete R920
16 x matching 16GB RDIMM
1 x RTX3060 12GB
5 x GTX1060 6GB
6 x Comino/true-x16 powered risers
1 x 1200W external GPU PSU
1 x MCX353A-class FDR NIC
1 x SATA SSD + caddy
1 x 6-GPU external frame
1 x FDR DAC (once switch/second node exists)
```

Then execute `VALIDATION.md`. Only the successful one-node and two-node gates justify the bulk order.