import os 
from dotenv import load_dotenv

from phi.agent import Agent 
from phi.model.groq import Groq 
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

GROQ_API_KEY= os.environ['GROQ_API_KEY']

web_search_agent = Agent(
    name="AI News LinkedIn Curator",
    role="Create a professional LinkedIn post about the latest AI development",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=[
        "Format news as compelling LinkedIn post",
        "Include 3-5 key AI news developments",
        "Write in a professional,engaging tone",
        "Use bullet points for redability",
        "Include relevent hashtag",
        "Provide source links for crediblity"
        "Highlights the broader impact of AI development",
        "End with an engagement prompt",
        "Ensure content is suitable for professional networking audience",
        "Keep the total post lenght under 3000 characters"
    
    ],
    show_tools_calls = True,
    markdown=True
)

# News Relevance Agent 
news_relevance_agent = Agent(
    name ="News Relevance Validator",
    role="Critically evaluate AI news for social media posting ",
    model=Groq(id="llama-3.3-70b-versatile"),
    instructions=[
        "Carefully assess the generated Ai news content",
        "Determine if the content is suitable for LinkedIn posting",
        """Check for:
        -Professionalism
        -Current relevance
        -Potential impact
        -Absence of controversial content""",
        """Provide a stuctured evaluation with:
        -Suitablity score (0-10)
        -Posting recommendation(Ye/No)
        -Specific reasons for evaluation"""
        "If not suitable ,explain specific reasons",
        "Suggest potential modification if needed",
        "Respond with 'No' in the psting recommendation if contentis not suitable"
    

    ],
    show_tool_calls=True,
    markdown=True
)

def main():
    #Generate AI news content
    news_response= web_search_agent.run("5 latest significant AI news developments with sources",stream=False)

    # Validation the generated news content
    validation_response= news_relevance_agent.run(
        f"Evaluate the following Ai content for Linkdin post suitability :\n\n{news_response.content}",
        stream=False
    )

    # Check if validation recommends not posting
    news_content = news_response.content
    if "<function=duckduckgo_news" in validation_response.content:
        news_content = ""
    else:
        news_content=news_response.content
    
    return {
        "news_content":news_content,
        "validation":validation_response.content
    }

if __name__ == '__main__':
    result = main()
    print("Generate News:")
    print(result['news_content'])
    print('\n validation result')
    print(result['validation'])