import os

import shap
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


class SHAPAnalyzer:
    def __init__(self, output_dir="results/shap"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def analyze_model(self, model, X, feature_names, model_name, label_type, plot_types):
        """
        执行SHAP分析并生成可视化图表
        """
        try:
            # 确保输入维度正确
            print(f"输入数据形状 ({model_name}):", X.shape)

            # 创建SHAP解释器
            if model_name == "RandomForest":
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif model_name in ["LSTM", "RNN", "MLP"]:
                model.eval()
                # 将NumPy数组转换为PyTorch张量
                if not isinstance(X, torch.Tensor):
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                else:
                    X_tensor = X
                # 将张量移动到正确的设备（GPU或CPU）
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_tensor = X_tensor.to(device)
                model = model.to(device)
                explainer = shap.DeepExplainer(model, X_tensor)
                shap_values = explainer.shap_values(X_tensor, check_additivity=False)
            else:
                # 对于其他模型，使用KernelExplainer并抽样背景数据集
                background_size = min(50, X.shape[0])
                background = shap.sample(X, background_size) if X.shape[0] > 1 else X
                explainer = shap.KernelExplainer(model.predict_proba, background)
                # 计算SHAP值
                shap_values = explainer.shap_values(X)

            print(f"原始SHAP值形状 ({model_name}):",
                  [sv.shape if hasattr(sv, 'shape') else f"list({len(sv)})"
                   for sv in shap_values] if isinstance(shap_values, list) else shap_values.shape)

            # 处理SHAP值格式
            if model_name == "RandomForest":
                # 树模型的SHAP值是一个包含两个元素的列表 [负类_SHAP, 正类_SHAP]
                # 取正类SHAP值（索引1）
                shap_values_pos = shap_values
                print(f"处理后SHAP值形状 ({model_name}):", shap_values_pos.shape)
            elif isinstance(shap_values, list):
                # 其他模型返回列表的情况
                shap_values_pos = shap_values[0]  # 通常第一个元素是主输出
                if len(shap_values) > 1 and hasattr(shap_values[1], 'shape'):
                    shap_values_pos = shap_values[1]  # 尝试取正类
            else:
                shap_values_pos = shap_values
                if model_name in ["RNN", "LSTM"]:
                    shap_values_pos = shap_values_pos.reshape(shap_values_pos.shape[0], -1)
                    X = X.reshape(shap_values_pos.shape[0], -1)

            # 确保SHAP值是二维的
            if len(shap_values_pos.shape) == 3 and shap_values_pos.shape[2] == 2:
                # 如果是三维数据 (样本, 特征, 类别)，取正类
                shap_values_pos = shap_values_pos[:, :, 1]
            elif len(shap_values_pos.shape) == 3 and shap_values_pos.shape[2] == 1:
                shap_values_pos = shap_values_pos[:, :, 0]

            print(f"最终SHAP值形状 ({model_name}):", shap_values_pos.shape)

            # 生成图表
            self.generate_plots(shap_values_pos, X, feature_names, model_name, label_type, plot_types)

        except Exception as e:
            print(f"SHAP分析失败 ({model_name}): {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整堆栈跟踪

    def generate_plots(self, shap_values, X, feature_names, model_name, label_type, plot_types):
        """生成各种SHAP可视化图表"""
        # 确保SHAP值和X的样本数一致
        assert shap_values.shape[0] == X.shape[0], \
            f"SHAP值样本数({shap_values.shape[0]})与输入数据样本数({X.shape[0]})不匹配"

        # 确保SHAP值是二维的
        if len(shap_values.shape) != 2:
            print(f"警告: SHAP值维度为{len(shap_values.shape)}D，尝试展平")
            shap_values = shap_values.reshape(shap_values.shape[0], -1)

        # 确保特征数一致
        if shap_values.shape[1] != len(feature_names):
            print(f"警告: SHAP值特征数({shap_values.shape[1]})与特征名数量({len(feature_names)})不匹配")
            # 使用通用特征名
            feature_names = [f'feature_{i}' for i in range(shap_values.shape[1])]

        output_dir = os.path.join(self.output_dir, label_type)
        os.makedirs(output_dir, exist_ok=True)
        # 特征重要性条形图
        if "bar" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance_bar.png"))
            plt.show()
            plt.close()

        # 蜂群图
        if "beeswarm" in plot_types:
            plt.figure()
            shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance_beeswarm.png"))
            plt.show()
            plt.close()

        # 单个特征依赖图
        if "dependence" in plot_types:
            for i, feature in tqdm(enumerate(feature_names), desc="生成依赖图", total=len(feature_names)):
                plt.figure()
                shap.dependence_plot(i, shap_values, X, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"dependence_{feature}.png"))
                plt.show()
                plt.close()