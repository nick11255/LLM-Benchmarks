
import csv
import matplotlib.pyplot as plt


batch, thru, lat = [], [], []
with open("results.csv") as f:
    for r in csv.DictReader(f):
        batch.append(int(r["batch_size"]))
        thru.append(float(r["throughput_tok_s"]))
        lat.append(float(r["latency_s"]))

# throughput vs batch size 
plt.figure()
plt.plot(batch, thru, marker="o")
plt.xlabel("batch size")
plt.ylabel("throughput (tokens/sec)")
plt.title("Throughput scales with batch size")
plt.xscale("log", base=2)          
plt.grid(True, alpha=0.3)
plt.savefig("throughput.png", dpi=120, bbox_inches="tight")

#latency vs throughput
plt.figure()
plt.plot(thru, lat, marker="o")
for b, x, y in zip(batch, thru, lat):
    plt.annotate(f"b={b}", (x, y), fontsize=8,
                 textcoords="offset points", xytext=(5, 5))
plt.xlabel("throughput (tokens/sec)")
plt.ylabel("latency (seconds)")
plt.title("Latency vs throughput (static batching)")
plt.grid(True, alpha=0.3)
plt.savefig("pareto.png", dpi=120, bbox_inches="tight")
