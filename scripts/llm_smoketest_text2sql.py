import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gpt2xl_agent import GPT2XLAgent

def main():
    schema = """Tables:
students(id, name, age)
courses(id, title)
enrollments(student_id, course_id)
"""
    question = "How many students are there?"

    agent = GPT2XLAgent()
    sql = agent.generate_sql(schema, question, max_new_tokens=80)

    print("Generated SQL:")
    print(sql)

if __name__ == "__main__":
    main()
