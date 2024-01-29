from acs import VectorStore
from pprint import pprint
# Perform a similarity search
query = "RoleInstance in Region [East US] for RoleInstance [bl-azurestackhci-dp-prod] is unhealthy. Monitor [RoleInstanceHealthy] in the account [azurestackhci] evaluated to true"
query = "tell me a joke"
query = "ARM operation [ARMUpdateCluster] in [East US] is failing. Monitor [ARMOperationWorking] in the account [azurestackhci] evaluated to true"
query = "langchain"
# docs = VectorStore.similarity_search(
#     query= query,
#     k=3,
#     search_type="similarity",
# )
docs = VectorStore.similarity_search_with_relevance_scores(
    query= query,
    k=100,
    score_threshold=0.1
)
pprint(docs)
print("---------------------------------")
count = 0
for doc, score in docs:
   # print(doc.page_content)
    print(count)
    print(doc.metadata)
    print(score)
    count += 1
    print("---")