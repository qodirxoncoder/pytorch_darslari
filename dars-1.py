import torch

# 1. Oddiy 1 o'lchamli tensor (Vektor)
vector = torch.tensor([1.0, 2.0, 3.0])
print("Vektor shakli:", vector.shape) # torch.Size([3])

# 2. 2 o'lchamli tensor (Matritsa - jadval ko'rinishida)
# Bu ko'pincha: [Batch_size, Features] ko'rinishida bo'ladi
matrix = torch.tensor([[1, 2, 3], 
                       [4, 5, 6]])
print("Matritsa shakli:", matrix.shape) # torch.Size([2, 3])

# 3. Tasodifiy sonlardan iborat 3 o'lchamli tensor
# Bu ko'pincha matnlar uchun ishlatiladi: [Batch_size, Sequence_Length, Embedding_Size]
text_tensor = torch.randn(32, 10, 64) 
print("Matn tensori shakli:", text_tensor.shape)


"""2-qadam: Matematik amallar va Matritsalar ko'paytmasi"""


# A matritsa: 2 ta satr, 3 ta ustun
A = torch.tensor([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])

# B matritsa: 3 ta satr, 2 ta ustun
B = torch.tensor([[7.0, 8.0], 
                  [9.0, 10.0], 
                  [11.0, 12.0]])

# Matritsalarni ko'paytirish
C = A @ B  # yoki torch.matmul(A, B)

print("Natija:\n", C)
print("Yangi shakli:", C.shape) # Natija: torch.Size([2, 2])