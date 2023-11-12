(critical) need to create an empty folder ./data/alpha_dataset/ on your own!

dataset.py 				由原始因子和股票数据得到dataset
model.py 				搭建模型
loss.py 				自定义损失函数
run.py					训练和测试，可重新选取模型、损失函数、优化器
main.py					可直接在终端运行的主函数，可在config中修改参数

./data/alpha_list/			存放原始因子数据
./data/HS300_close.pkl			沪深300成分股收盘价
./data/alpha_dataset/			存放因子dataset，即同一时间段生成一次dataset后无需重复生成

./models/20xxxxxx_20yyyyyy/n.pkl	在某一时间段上训练，第n个epoch得到的模型
./models/20xxxxxx_20yyyyyy/preds/	存放在某一时间段上训练得到最好模型的测试结果
