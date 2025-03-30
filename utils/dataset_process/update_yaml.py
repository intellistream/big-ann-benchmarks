import os
import yaml


def update_yaml_configs(folder_path, new_dataset_name):
    """遍历文件夹，读取 config.yaml 并添加新的数据集配置，保持原格式"""
    for algo_folder in os.listdir(folder_path):
        algo_path = os.path.join(folder_path, algo_folder)
        config_file = os.path.join(algo_path, 'config.yaml')

        if not os.path.isdir(algo_path) or not os.path.exists(config_file):
            continue

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        if 'yfcc-10M' not in config_data:
            print(f"警告: {config_file} 中未找到 yfcc-10M 配置，跳过。")
            continue

        # 复制 yfcc-10M 的配置信息并修改名称
        new_config = config_data['yfcc-10M']
        config_data[new_dataset_name] = new_config

        # 写回 YAML 文件，确保格式与原来一致
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False, width=1000)

        print(f"已更新 {config_file}，添加 {new_dataset_name} 配置。")


# 示例调用
update_yaml_configs('../../neurips23/filter', 'Youtube-rgb')
