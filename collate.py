import random
import torch

truncate_len = 45


def collate_fn(batch_data):
    user_ids, item_ids, labels = [], [], []
    u_items, u_users, u_users_items, i_users = [], [], [], []
    u_items_len, u_users_len, i_users_len = [], [], []

    for data, u_items_u, u_users_u, u_users_items_u, i_users_i in batch_data:

        (uid, iid, label) = data
        user_ids.append(uid)
        item_ids.append(iid)
        labels.append(label)

        # user-items
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))

        # user-users and user-users-items
        if len(u_users_u) <= truncate_len:
            u_users.append(u_users_u)
            u_u_items = []
            for uui in u_users_items_u:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)
        else:
            sample_index = random.sample(list(range(len(u_users_u))), truncate_len)
            u_users.append([u_users_u[si] for si in sample_index])

            u_users_items_u_tr = [u_users_items_u[si] for si in sample_index]
            u_u_items = []
            for uui in u_users_items_u_tr:
                if len(uui) < truncate_len:
                    u_u_items.append(uui)
                else:
                    u_u_items.append(random.sample(uui, truncate_len))
            u_users_items.append(u_u_items)

        u_users_len.append(min(len(u_users_u), truncate_len))

        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    u_users_maxlen = max(u_users_len)
    i_users_maxlen = max(i_users_len)

    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)

    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)

    u_user_item_pad = torch.zeros([batch_size, u_users_maxlen, u_items_maxlen, 2], dtype=torch.long)
    for i, uu_items in enumerate(u_users_items):
        for j, ui in enumerate(uu_items):
            u_user_item_pad[i, j, :len(ui), :] = torch.LongTensor(ui)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    user_ids = torch.LongTensor(user_ids)
    item_ids = torch.LongTensor(item_ids)
    labels = torch.FloatTensor(labels)

    return user_ids, item_ids, labels, u_item_pad, u_user_pad, u_user_item_pad, i_user_pad
