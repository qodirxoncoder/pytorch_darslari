# LLM (Katta til modellari) dunyosiga o‘tamiz!

import torch
import torch.nn as nn

# 1. Biz o'qitmoqchi bo'lgan kichik matn (Dataset)
matn = "men backend dasturchiman va men sunniy intellektni urganaman"

# 2. Matnni so'zlarga bo'lib chiqamiz va takrorlanmas so'zlar ro'yxatini (Lug'at) tuzamiz
sozlar = sorted(list(set(matn.split())))
lugat_hajmi = len(sozlar)

print("Bizning lug'at:", sozlar)
print("Lug'atdagi so'zlar soni:", lugat_hajmi)

# 3. Backend dasturchi sifatida sizga juda tanish bo'lgan Map (Dictionary) tuzamiz
# So'zdan raqamga va raqamdan so'zga o'girgichlar
soz_to_id = {soz: i for i, soz in enumerate(sozlar)}
id_to_soz = {i: soz for i, soz in enumerate(sozlar)}

print("\nSo'zlarning ID raqamlari:", soz_to_id)

# 4. "men sunniy dasturchiman" degan gapni raqamlarga o'giramiz (Tokenization)
sinov_gapi = "men sunniy dasturchiman"
tokenlar = [soz_to_id[s] for s in sinov_gapi.split()]
print(f"\n'{sinov_gapi}' gapi raqamlarda:", tokenlar)

# 5. EMBEDDING QATLAMY (Sehrli qatlam)
# Har bir raqamli token uchun 4 ta raqamdan iborat koordinata (vektor) ajratamiz
# nn.Embedding(lug'at_hajmi, vektor_o'lchami)
embedding_qatlami = nn.Embedding(num_embeddings=lugat_hajmi, embedding_dim=4)

# Tokenlarni PyTorch tensoriga o'giramiz va Embeddingdan o'tkazamiz
token_tensor = torch.tensor(tokenlar)
vektorlar = embedding_qatlami(token_tensor)

print("\nHar bir so'zning 4 o'lchamli fazodagi vektor koordinatalari:\n", vektorlar)
print("Vektor shakli:", vektorlar.shape) # torch.Size([3, 4]) -> 3 ta so'z, har biri 4 o'lchamli