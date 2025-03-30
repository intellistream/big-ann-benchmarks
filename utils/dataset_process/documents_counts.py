def count_downloaded_files(log_file):
    """统计成功下载的文件数量"""
    count = 0
    with open(log_file, "r", encoding="utf-8") as file:
        for line in file:
            if "Successfully downloaded" in line:
                count += 1
    return count

# 运行程序
log_file_path = "counts.txt"  # 替换为你的日志文件路径
downloaded_count = count_downloaded_files(log_file_path)

print(f"成功下载的文件数量: {downloaded_count}")
