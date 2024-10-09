先测单任务:

Task1.py ---猜测物体类型 python Task1.py --batch_size 200 --seed 43
    图像处理 ceiling_pre/ml/model.pth
                -----实验结果：陷入局部最小值0.3+
            ceiling_resnet/model.pth
                -----五千多epoch测试集准确率0.7+，验证集0.5左右
            ceiling_pre/model.pth
                -----验证集准确率一直是0.336（应该是都预测为同一类了）
            fd_b/cf/cf_model.pth
                -----0.5左右
    原因：不能直接冻结参数！要微调！

    微调后，图像处理用ceiling_resnet/model.pth，30多个epoch后就到达了最佳性能，结果在Multi-Task/mt1/SpotDiff.png

Task2.py ---猜测物体颜色
    在100个epoch后收敛，结果在Multi-Task/mt2/results.png

    问题：不到0.6的准确率。。。

多任务！

没用cross-attention,用的还是差值,也就是说全共享参数，除了最后的分类层不同
MulTask.py --猜类别和猜颜色
    结果还是猜颜色只有0.45左右 猜类别有0.9
    在50epoch后收敛，


用cross-attention后，两个任务都只能分类到一类中