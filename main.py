import torch
import torch_geometric.transforms as T

from db_transformer import DBTransformer
from db_transformer.data import DataLoader

torch.manual_seed(1)


def train():
    target_table = "molecule"

    print("Getting tables")
    data_loader = DataLoader("./dataset/mutag")
    table_data, hetero_data, labels = data_loader.load(target_table)

    epochs = 1000

    dim = 32
    dim_out = 4
    heads = 2
    layers = 4
    attn_dropout = 0.1
    ff_dropout = 0.1


    hetero_data = T.ToUndirected()(hetero_data)
    hetero_data = T.AddSelfLoops()(hetero_data)

    transformer = DBTransformer(hetero_data.metadata(), dim, dim_out, heads, attn_dropout, ff_dropout, table_data, layers)

    print(transformer)

    optim = torch.optim.Adam(transformer.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(epochs):
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
