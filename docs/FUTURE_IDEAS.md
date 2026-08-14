






- Handle mixed-scope questions (BNS + constitutional law)
- Higher disclaimer strength for time-sensitive/high-stakes questions  
- Golden dataset category: mixed_scope_high_stakes




# for handling the  ambigious message? 
# also there will be the chat history ans question like having previous context try to better rewrite query for this
#or handle this case

QUERY_REWRITE_PROMPT = """Rewrite the user's question into a short, clear query using formal Indian legal terminology, suitable for searching a legal database. Do not answer the question. Output ONLY the rewritten query, nothing else.

Example 1:
User question: can i go to jail for taking his phone without asking
Rewritten query: punishment for theft of movable property

Example 2:
User question: is it ok to hit someone if they hit me first
Rewritten query: right of private defence against bodily harm

Example 3:
User question: what happens if i lie to police
Rewritten query: punishment for giving false information to a public servant

User question: {question}
Rewritten query:"""