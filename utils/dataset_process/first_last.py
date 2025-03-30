input_file = "../../data/YouTube-rgb/filter/query_metadata.txt"
output_file = "../../data/YouTube-rgb/filter/query_metadata_1.txt"

with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out_f:
    for line in f:
        tags = line.strip().split(",")  # 以逗号分隔标签
        tags = [t.strip() for t in tags if t.strip().isdigit()]  # 只保留数字

        if len(tags) == 0:
            continue  # 跳过空行
        elif len(tags) > 2:
            tags = [tags[0], tags[-1]]  # 取第一个和最后一个

        out_f.write(",".join(tags) + "\n")

print(f"已修复 query_metadata.txt，保存为 {output_file}")
