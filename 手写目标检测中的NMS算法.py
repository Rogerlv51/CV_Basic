## 注意NMS仅针对单一类别去去除多余的框，比如我们现在有一个预测出是人脸框的6个框，参数表示为(x1,y1,x2,y2,scores)采用左上右下坐标
## 假设我们得到了六个框A：0.9，B：0.85，C：0.7，D：0.6，E：0.4，F：0.1
## 下面展示算法流程：(1) 取出最大置信度的那个目标框 A 保存下来 (2) 分别判断 B-F 这５个目标框与 A 的重叠度 IOU ，
## 如果IOU大于我们预设的阈值（一般为 0.5），则将该目标框丢弃。假设此时丢弃的是C和F两个目标框，这时候该序列中只剩下BDE这三个。 
## (3) 重复以上流程，直至排序序列为空。

import numpy as np
def nms(boxes, scores, thresh=0.5):
    # bboxees维度为 [N, 4]，scores维度为 [N, 1]，均为np.array()
    x1 = boxes[:, 0] # 取出所有框的左上角x坐标
    y1 = boxes[:, 1] # 取出所有框的左上角y坐标
    x2 = boxes[:, 2] # 取出所有框的右下角x坐标
    y2 = boxes[:, 3] # 取出所有框的右下角y坐标

    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 计算所有框的面积，方便后面计算IOU
    order = scores.argsort()[::-1] # 按置信度从大到小排序，返回索引值
    keep = []   # 用于保存框的索引

    while len(order)>0 :
        if len(order) == 1:   # 如果只有一个框，直接保存跳出
            index = order[0]
            keep.append(index)
            break
        else:
            index = order[0]
            keep.append(index)  # 保存置信度最大的框的索引
        
        # 计算置信度最大的框与其他框的IOU
        xx1 = np.maximum(x1[index], x1[order[1:]])   # 这会得到一个交叉区域左上坐标的list，后面同理
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])

        # 计算交叉部分面积
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)

        iou = w*h / (areas[index] + areas[order[1:]] - w*h) # 计算IOU
        
        ind =  np.where(iou<=thresh)[0]   # 小于等于阈值的索引则保留下来
        if len(ind)==0:
            break
        order = order[ind+1]    # 更新order，继续循环
    return keep

if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210], [250, 250, 420, 420], [220, 220, 320, 330], [100, 100, 210, 210], [230, 240, 325, 330], [500, 500, 520, 530]])
    scores = np.array([0.4, 0.6, 0.92, 0.72, 0.1, 0.88])
    keep = nms(boxes, scores)
    print(keep) 