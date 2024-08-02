import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from MulticoreTSNE import MulticoreTSNE as TSNE


CLASS2COLOR = {
    0: (0, 0, 0),
    1: (255, 124, 116),
    2: (223, 120, 242),
    3: (0, 176, 0),
    4: (0, 108, 240),
    5: (239, 209, 207),
    6: (167, 245, 155),
    7: (255, 113, 165),
    8: (255, 172, 168),
    9: (235, 168, 233),
    10: (209, 209, 209),
}


class RunTsne():
    def __init__(self,
                 dataset_name,        # 数据集名称
                 num_class,           # 类的个数
                 output_dir='./',     # 保存的路径
                 extention='.png',    # 保存图片的格式
                 duplication=10):     # 程序循环运行几次，即保存多少张结果图片
        
        
                # selected_cls,        # 选择可视化几个类别
                # domId2name,          # 不同域的ID
                # trainId2name,        # 标签中每个ID所对应的类别
                # trainId2color=None,  # 标签中每个ID所对应的颜色

        self.dataset_name = dataset_name
        self.num_class = num_class
        
        selected_cls, domId2name, trainId2name, trainId2color = self.init_dataset(dataset_name)
        
        self.domId2name = domId2name
        self.name2domId = {v:k for k,v in domId2name.items()}
        self.trainId2name = trainId2name
        self.trainId2color = trainId2color
        self.selected_cls = selected_cls
        self.name2trainId = {v:k for k,v in trainId2name.items()}
        self.selected_clsid = [self.name2trainId[x] for x in selected_cls]

        self.tsne_path = output_dir
        os.makedirs(self.tsne_path, exist_ok=True)
        self.extention = extention
        self.duplication = duplication
        
        self.init_basket()    # 初始化

        self.max_pointnum = 10200
        self.perplexity = 50
        self.learning_rate = 4800
        self.n_iter = 3000
        self.TSNE = TSNE(n_components=2, perplexity=self.perplexity, learning_rate=self.learning_rate,
                            n_iter=self.n_iter, verbose=1, n_jobs=4)

    def init_dataset(self, dataset_name):
        if dataset_name == 'bhx_sammed':
            # 自己指定要进行 t-SNE 的类别（可以根据 t-SNE 的效果选择最好的几个类别即可）
            selected_cls = ['right', 'left', 'third', 'fourth'] 
            # 为每个数据集指定一个 ID
            domId2name = {0: dataset_name}
            # 标签映射到类别名
            trainId2name = {0:'bg', 1: 'right', 2: 'left', 3: 'third', 4: 'fourth'}
            # 标签映射到颜色
            trainId2color = {i: CLASS2COLOR[i] for i in range(5)}
        elif dataset_name == 'sabs_sammed' or dataset_name == 'sabs_sammed_roi':
            # 自己指定要进行 t-SNE 的类别（可以根据 t-SNE 的效果选择最好的几个类别即可）
            selected_cls = ['spleen', 'rkid', 'lkid', 'gall', 'liver', 'sto', 'aorta', 'panc'] 
            # 为每个数据集指定一个 ID
            domId2name = {0: dataset_name}
            # 标签映射到类别名
            trainId2name = {0:'bg', 1: 'spleen', 2: 'rkid', 3: 'lkid', 4: 'gall', 5: 'liver', 6: 'sto', 7: 'aorta', 8: 'panc'}
            # 标签映射到颜色
            trainId2color = {i: CLASS2COLOR[i] for i in range(9)}
        
        return selected_cls, domId2name, trainId2name, trainId2color
        
    def init_basket(self):
        self.feat_vecs = torch.tensor([]).cuda()            # 特征向量
        self.feat_vec_labels = torch.tensor([]).cuda()      # 特征向量的类别
        self.feat_vec_domlabels = torch.tensor([]).cuda()   # 特征向量的域信息
        self.mem_vecs = None                                # 聚类中心的向量
        self.mem_vec_labels = None                          # 聚类中心的类别

    def input_memory_item(self,m_items):
        self.mem_vecs = m_items[self.selected_clsid]
        self.mem_vec_labels = torch.tensor(self.selected_clsid).unsqueeze(dim=1).squeeze()

    def input2basket(self, feature_map, gt_cuda, datasetname):
        b, c, h, w = feature_map.shape # [b, 256, 32, 32]
        features = F.normalize(feature_map.clone(), dim=1) # [b, 256, 32, 32]
        gt_cuda = gt_cuda.clone()
        H, W = gt_cuda.size()[-2:]
        gt_cuda = F.one_hot(gt_cuda, num_classes=self.num_class+1)

        gt = gt_cuda.view(b, -1, self.num_class + 1)
        denominator = gt.sum(1).unsqueeze(dim=1)
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze()

        features = F.interpolate(features, [H, W], mode='bilinear', align_corners=True)
        # 这里是将feature采样到跟标签一样的大小。当然也可以将标签采样到跟feature一样的大小
        features = features.view(b, c, -1)
        nominator = torch.matmul(features, gt.type(torch.float32))
        nominator = torch.t(nominator.sum(0))  # batchwise sum

        for slot in self.selected_clsid:
            if denominator[slot] != 0:
                cls_vec = nominator[slot] / denominator[slot]  # mean vector
                cls_label = (torch.zeros(1, 1) + slot).cuda()
                dom_label = (torch.zeros(1, 1) + self.name2domId[datasetname]).cuda()
                self.feat_vecs = torch.cat((self.feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0)
                self.feat_vec_labels = torch.cat((self.feat_vec_labels, cls_label), dim=0)
                self.feat_vec_domlabels = torch.cat((self.feat_vec_domlabels, dom_label), dim=0)

    def draw_tsne(self, domains2draw, adding_name=None, plot_memory=False, clscolor=True):
        feat_vecs_temp = F.normalize(self.feat_vecs.clone(), dim=1).cpu().numpy()
        feat_vec_labels_temp = self.feat_vec_labels.clone().to(torch.int64).squeeze().cpu().numpy()
        feat_vec_domlabels_temp = self.feat_vec_domlabels.clone().to(torch.int64).squeeze().cpu().numpy()

        if self.mem_vecs is not None and plot_memory:
            mem_vecs_temp = self.mem_vecs.clone().cpu().numpy()
            mem_vec_labels_temp = self.mem_vec_labels.clone().cpu().numpy()

        if adding_name is not None:
            tsne_file_name = adding_name+'_feature_tsne_among_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)
        else:
            tsne_file_name = 'feature_tsne_among_' + ''.join(domains2draw) + '_' + str(self.perplexity) + '_' + str(self.learning_rate)
        tsne_file_name = os.path.join(self.tsne_path,tsne_file_name)

        if clscolor:
            sequence_of_colors = np.array([list(self.trainId2color[x]) for x in range(self.num_class+1)])/255.0
        else:
            sequence_of_colors = ["tab:purple", "tab:pink", "lightgray","dimgray","yellow","tab:brown","tab:orange","blue","tab:green","darkslategray","tab:cyan","tab:red","lime","tab:blue","navy","tab:olive","blueviolet", "deeppink","red"]
            sequence_of_colors[1] = "tab:olive"
            sequence_of_colors[2] = "tab:grey"
            sequence_of_colors[5] = "tab:cyan"
            sequence_of_colors[8] =  "tab:pink"
            sequence_of_colors[10] = "tab:brown"
            sequence_of_colors[13] = "tab:red"

        name2domId = {self.domId2name[x] : x for x in self.domId2name.keys()}
        domIds2draw = [name2domId[x] for x in domains2draw]
        name2trainId = {v:k for k,v in self.trainId2name.items()}
        trainIds2draw = [name2trainId[x] for x in self.selected_cls]
        domain_color = ["tab:blue", "tab:green","tab:orange","tab:purple","black"]
        assert len(feat_vec_domlabels_temp.shape) == 1
        assert len(feat_vecs_temp.shape) == 2
        assert len(feat_vec_labels_temp.shape) == 1

        # domain spliting
        dom_idx = np.array([x in domIds2draw for x in feat_vec_domlabels_temp])
        feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], \
                                                                       feat_vec_domlabels_temp[dom_idx]

        # max_pointnum random sampling.
        if feat_vecs_temp.shape[0] > self.max_pointnum:
            pointnum_predraw = feat_vec_labels_temp.shape[0]
            dom_idx = np.random.randint(0,pointnum_predraw,self.max_pointnum)
            feat_vecs_temp, feat_vec_labels_temp, feat_vec_domlabels_temp = feat_vecs_temp[dom_idx, :], feat_vec_labels_temp[dom_idx], feat_vec_domlabels_temp[dom_idx]

        if self.mem_vecs is not None and plot_memory:
            mem_address = feat_vecs_temp.shape[0]
            vecs2tsne = np.concatenate((feat_vecs_temp,mem_vecs_temp))
        else:
            vecs2tsne = feat_vecs_temp

        for tries in range(self.duplication):
            X_embedded = self.TSNE.fit_transform(vecs2tsne)
            print('\ntsne done')
            X_embedded[:,0] = (X_embedded[:,0] - X_embedded[:,0].min()) / (X_embedded[:,0].max() - X_embedded[:,0].min())
            X_embedded[:,1] = (X_embedded[:,1] - X_embedded[:,1].min()) / (X_embedded[:,1].max() - X_embedded[:,1].min())

            if self.mem_vecs is not None and plot_memory:
                feat_coords = X_embedded[:mem_address,:]
                mem_coords = X_embedded[mem_address:,:]
            else:
                feat_coords = X_embedded

            ##### color means class
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            for dom_i in domIds2draw:
                for cls_i in trainIds2draw:
                    temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
                    ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                               color=sequence_of_colors[cls_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=20, marker = 'x')

            if self.mem_vecs is not None and plot_memory:
                for cls_i in trainIds2draw:
                    ax.scatter(mem_coords[mem_vec_labels_temp == cls_i, 0], mem_coords[mem_vec_labels_temp == cls_i, 1],
                               color=sequence_of_colors[cls_i], label='mem_' + str(self.trainId2name[cls_i]), s=100, marker="^",edgecolors = 'black')

            print('scatter plot done')
            lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            tsne_file_path = tsne_file_name+'_'+str(tries)+'_colorclass'+self.extention
            fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            # plt.show()
            fig.clf()

            ##### color means domains
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)

            for dom_i in domIds2draw:
                for cls_i in trainIds2draw:
                    temp_coords = feat_coords[(feat_vec_labels_temp == cls_i) & (feat_vec_domlabels_temp == dom_i),:]
                    ax.scatter(temp_coords[:, 0], temp_coords[:, 1],
                               color= domain_color[dom_i], label=self.domId2name[dom_i]+'_'+self.trainId2name[cls_i], s=20, marker = 'x')

            if self.mem_vecs is not None and plot_memory:
                for cls_i in trainIds2draw:
                    ax.scatter(mem_coords[mem_vec_labels_temp == cls_i, 0], mem_coords[mem_vec_labels_temp == cls_i, 1],
                               color=sequence_of_colors[cls_i], label='mem_' + str(self.trainId2name[cls_i]), s=100, marker="^",edgecolors = 'black')

            print('scatter plot done')
            lgd = ax.legend(loc='upper center', bbox_to_anchor=(1.15, 1))
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            tsne_file_path = tsne_file_name+'_'+str(tries)+'_colordomain'+self.extention
            fig.savefig(tsne_file_path, bbox_extra_artists=(lgd,), bbox_inches='tight')
            # plt.show()
            fig.clf()

            # print memory coordinate
            if self.mem_vecs is not None and plot_memory:
                print("memory coordinates")
                for i,x in enumerate(mem_vec_labels_temp):
                    print(mem_coords[i,:],self.trainId2name[x])
        return tsne_file_path
