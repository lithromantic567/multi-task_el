import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10, help='batch的大小')
    parser.add_argument('--seed', type=int, default=0, help='随机种子的值')
    #parser.add_argument('--train', type=str, default='output_train.txt', help='输出文件的路径')
    #parser.add_argument('--eval', type=str, default='output_eval.txt', help='输出文件的路径')
    parser.add_argument('--model', type=str, default='model.pth', help='输出文件的路径')
    parser.add_argument('--fig', type=str, default='output_fig',help='输出图片的名称')
    parser.add_argument('--title',type=str, default='fig_title',help='输出图片的标题')
    #parser.add_argument('--reward', type=str, default='reward.txt', help='输出奖励的路径')
    #parser.add_argument('--loss', type=str, default='loss.txt', help='输出损失的路径')
    parser.add_argument('--py',type=str, default='train',help='运行的文件')
    parser.add_argument('--ylabel',type=str, default='Accuracy',help='纵坐标')
    parser.add_argument('--cuda',type=str, default="cuda:1",help='cuda')


    args = parser.parse_args()
    return args
