"""
MAIN APP (terminal) — Upload a PDF, ask questions in normal human language.

Examples:
  python pdf_rag_app.py --pdf "C:\\path\\policy.pdf"
  python pdf_rag_app.py --pdf policy.pdf --question "What is the waiting period for maternity?"
  python pdf_rag_app.py --pdf policy.pdf --rebuild      (re-embed if you replaced the PDF)

First run indexes the PDF (takes a bit for 100+ pages, costs a little embedding credit).
Next runs reuse the saved index instantly. Type 'quit' to exit.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_core import PdfChat, get_vector_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Ask questions about any PDF (even 100+ pages) in plain English.")
    ap.add_argument("--pdf", required=True, help="Path to your PDF file")
    ap.add_argument("--question", help="Ask one question and exit (optional)")
    ap.add_argument("--rebuild", action="store_true", help="Force re-indexing of the PDF")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and put your key in it.")
        return

    print(f"Indexing: {args.pdf} ...")
    store, n_pages, n_chunks, reused = get_vector_store(args.pdf, force_rebuild=args.rebuild)
    if reused:
        print(f"Reused saved index ({n_chunks} chunks already embedded). No re-embedding cost.")
    else:
        print(f"Indexed {n_pages} pages -> {n_chunks} chunks.")

    # Give PdfChat this PDF's chunks so BM25 (hybrid search) can run too
    from rag_core import load_and_split_pdf
    _, _chunks = load_and_split_pdf(args.pdf)
    chat = PdfChat(store, docs=_chunks)

    if args.question:
        answer, pages, _ = chat.ask(args.question)
        print(f"\nQ: {args.question}\nA: {answer}\n(from pages: {pages})")
        return

    print("\nAsk me anything about this PDF. Type 'quit' to exit.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not q:
            continue
        answer, pages, _ = chat.ask(q)
        print(f"\nAnswer: {answer}")
        print(f"(based on pages: {pages})")


if __name__ == "__main__":
    main()
