def print_conv_history(issue,verbose=True):
    result=""
    request=issue["fields"]["description"]
    result += f"Request: {request}\n"
    result += "------ Conversation History ------\n"
    result += "------ Conversation History ------"
    if verbose:
        print(f"Request: {request}")
        print("----------------")
        print("------ Conversation History ------")
    for comment in issue["fields"]["comment"]["comments"]:
        body= comment["body"]
        author_name= comment["author"]["displayName"]
        if verbose:
            print(f"- Author: {author_name}")
            print(f"Message: {body}")
            print("#"* 40)
        result += f"Author: {author_name}\n"
        result += f"Message: {body}\n"
        result += "#"* 40 + "\n"
    if verbose:
        print("----------------")
    return result
     

prompt_template="""Here is the conversation history of a Jira issue. Please explain clearly what is the request,\
if the issue was solved and give a full, clear tutorial to solve the user's request.
Do not mention steps that were useless to solve the issue.
For example 'The technician assumed that following the instructions in the article would resolve the issue, and therefore did not provide further assistance.' or 'The technician marked the issue as resolved.' are useless to solve the issue.
Also, do not cite names and only cite the important steps to solve the request.\
Your answer should look like a tutorial. It is not a summary of the conversation. You must also say if the issue could have been solved without assistance,
meaning if the user could have found the solution to its issue by themself.\
For example, if someone request a license and an operator say: 'I created your license',\
it means the person who made the request could not have done it by himslef.
But if the technician provided a link to solve the request, it means the user can do everything by himself.
Your answer must look like this:
Request: <request>
Request title: <request title>
Issue solved: <yes/no>
Issue could have been solved without assistance: <yes/no>
Steps to solve the issue:
1. <step 1>
2. <step 2>
3. <step 3>
etc.
Here is the conversation history:
{conv}
Only answer with the format above, do not add any other text. Remember to not cite any name, if you mention the tehcnician who solved the issue, just say 'the technician'.\
If your need to cite users, just say 'users' but do not cite any names. Any pseudo or term that could involve a user identity must be replace by something else. FOr example, 'add the license to qdhu and BauORa' must be replaced by 'add the license to desired users'.
Also remember your answer should be a tutorial, not a summary of the conversation.
"""


def build_doc_from_issue(issue,llm):
    key= issue["key"]
    url=f"https://jira.yourcompany.com/servicedesk/customer/portal/7/{key}"
    title= issue["fields"]["summary"]
    conv=print_conv_history(issue,verbose=False)
    prompt = prompt_template.format(conv=conv)
    response = llm.invoke(prompt)
    res= response.content
    req=res.split("Request title: ")[0].strip()
    tuto=res.split("Steps to solve the issue:")[1].strip()
    return {
        "id": key,
        "url": url,
        "title": title,
        "request": req,
        "content": req + "\n" + tuto,
    }
    
    