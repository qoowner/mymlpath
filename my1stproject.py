from transformers import pipeline

text = "Marine biologists have discovered a previously unknown species of deep-sea creatures in the Pacific Ocean near the Mariana Trench. The newly found organisms, measuring only 2-3 centimeters in length, exhibit unique bioluminescent properties that help them survive in the extreme darkness of the ocean's depths.Dr. Sarah Johnson, the lead researcher, described the discovery as significant for our understanding of deep-sea biodiversity. The creatures, named Abyssal Luminis, have evolved special adaptations to survive in one of Earth's most inhospitable environments, where water pressure reaches over 1,000 atmospheres.The discovery was made using advanced remotely operated vehicles equipped with high-definition cameras. The research team spent three months studying the area and collecting specimens for analysis. Preliminary findings suggest these creatures may play an important role in the deep-sea ecosystem.This finding shows how much we still don't know about our oceans, said Dr. Michael Chen, a marine ecologist. We've explored less than 5% of the world's oceans, and discoveries like this remind us of the incredible diversity hidden beneath the waves.The research has been published in Marine Biology Research and is expected to influence future ocean conservation efforts. Environmental groups are calling for increased protection of deep-sea habitats, citing this discovery as evidence of the need for better ocean conservation policies."



summarizer = pipeline("summarization")
result = summarizer(text,
           min_length = 30,
           max_length = 70)

emotions = pipeline("sentiment-analysis")
result_e = emotions(text)

print(f"Brief summary: {result[0]['summary_text']}")
print(f"Emotional coloring: {result_e[0]['label']}")

que = input("Ask a question about the text:")

questions = pipeline("question-answering", model="deepset/roberta-base-squad2")
result_q = questions(
    question = que,
    context = text
)


print(f"Response: {result_q[0]['answer']}")
