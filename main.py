import torch
import torch_geometric.transforms as T

from db_transformer import DBTransformer, MyMLP, SimpleTableTransformer
from db_transformer.data import DataLoader
from db_transformer.gnn import MyHeteroGNN

from torchmetrics.classification import MulticlassAccuracy

torch.manual_seed(1)


# device = torch.device('cuda')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train():
    epochs = 1000

    dim = 32
    dim_out = 2
    heads = 1
    attn_dropout = 0.1
    ff_dropout = 0.1

    def my_mlp(dim_in, table):
        return MyMLP(
            dim=(table.num_continuous + len(table.categories) + len(table.keys) + 1) * dim_in,
        )

    def my_transformer(dim_in, table):
        return SimpleTableTransformer(
            dim=dim_in,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            table=table,
        )

    layers = 2
    label_index = -1
    target_table = "molecule"

    print("Getting tables")
    data_loader = DataLoader("./dataset/mutag", 200, layers, torch.long, 150) # 150) #19011)
    data_loader.load_data_loader("./dbs/mutag", target_table)

    print(data_loader.metadata[0])
    print("===")
    print(data_loader.metadata[1])

    def my_gsage(dim_in):
        return MyHeteroGNN(dim_in, dim_in, data_loader.metadata)

    transformer = DBTransformer(dim, dim_out, my_transformer, my_gsage, data_loader.tables, layers)

    # print(transformer)

    optim = torch.optim.Adam(transformer.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = torch.nn.MSELoss()

    for i in range(epochs):
        tloss = 0
        inst = 0
        for index in range(data_loader.len):
            table_data, hetero_data, labels = data_loader.load(index, target_table, label_index)
            inst += len(labels)

            # print(hetero_data)
            hetero_data = T.ToUndirected()(hetero_data)
            hetero_data = T.AddSelfLoops()(hetero_data)
            # metadata = hetero_data.metadata()

            x = transformer(table_data, hetero_data, target_table)

            loss = loss_fn(x, labels)
            # tloss += float(loss)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            optim.step()
        if i % 1 == 0 and i != 0:
            with torch.no_grad():
                tlabels = None
                tpred = None

                for index in range(data_loader.test_len):
                    table_data, hetero_data, labels = data_loader.load(index, target_table, label_index, False)

                    hetero_data = T.ToUndirected()(hetero_data)
                    hetero_data = T.AddSelfLoops()(hetero_data)

                    x = transformer(table_data, hetero_data, target_table)

                    if tlabels is None:
                        tlabels = labels
                        tpred = x
                    else:
                        tlabels = torch.concat((tlabels, labels))
                        tpred = torch.concat((tpred, x))
                metric = MulticlassAccuracy(num_classes=2)
                print("Acc", metric(tpred.squeeze(), tlabels))


train()
