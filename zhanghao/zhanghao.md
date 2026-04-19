
3.26部署完成最新版的格物平台，但存在渲染报错 3.27解决报错问题，并用激励函数进行训练

4.10根据分类的csv数据文件训练左勾拳动作，目前已训练两个数据集，并上传replay动作和训练之后的动作视频
https://github.com/user-attachments/assets/6367964c-cd73-4e1b-991a-be1a109c7cf2

https://github.com/user-attachments/assets/ac41d074-e0b3-4248-8457-aa771b0e4046
第一个数据集是训练800w步的结果，训练动作和replay动作基本一致

https://github.com/user-attachments/assets/1777cd27-5634-475c-bb78-b76929b2ba5a

https://github.com/user-attachments/assets/c0292fcb-8bd3-4277-ab8d-595894b824ad
第二个数据集只训练了600w步，实际训练效果和replay动作差距较大，之后继续训练，看是否是训练步数不够导致的

同时调整了奖励函数，原本的奖励函数是线性奖励，我将其改成指数奖励，并对模型进行训练

4.19更新了新的奖励函数训练右勾拳，训练了1600w步，但训练结果不理想，只能稳定打出一拳后就倒地了，目前考虑如何更改奖励函数
