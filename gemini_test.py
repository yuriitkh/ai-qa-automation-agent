from google import genai
import os


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input="Explain in one sentence what software testing is."
    )

    print(interaction.output_text)


if __name__ == "__main__":
    main()