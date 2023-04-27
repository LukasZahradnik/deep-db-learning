import torch
import torch_geometric
import torch_geometric.transforms as T

from db_transformer import DBTransformer, MyMLP, SimpleTableTransformer
from db_transformer.data import DataLoader

torch.manual_seed(1)




def train():
    epochs = 1000

    dim = 32
    dim_out = 9
    heads = 2
    layers = 4
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


    target_table = "target"

    print("Getting tables")
    data_loader = DataLoader("./dataset/mutag", 300)
    data_loader.load_data_loader("./dbs/f1", target_table)

    def my_gsage(dim_in):
        return torch_geometric.nn.to_hetero(torch_geometric.nn.GraphSAGE(dim_in, dim_in, 1), data_loader.metadata, aggr="mean")

    transformer = DBTransformer(dim, dim_out, my_mlp, my_gsage, data_loader.tables, layers)

    print(transformer)

    optim = torch.optim.Adam(transformer.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(epochs):
        for index in range(data_loader.len):
            table_data, hetero_data, labels = data_loader.load(index, target_table)

            hetero_data = T.ToUndirected()(hetero_data)
            hetero_data = T.AddSelfLoops()(hetero_data)
            metadata = hetero_data.metadata()

            x = transformer(table_data, hetero_data, target_table)

            loss = loss_fn(x, labels)
            print(i, "Loss: ", loss)

            optim.zero_grad(set_to_none=True)
            loss.backward()

            optim.step()

            if i % 10 == 0:
                s = 0
                lab = torch.argmax(x, dim=1)
                for r, p in zip(labels, lab):
                    if r == p:
                        s += 1
                print(s, len(labels))


train()
