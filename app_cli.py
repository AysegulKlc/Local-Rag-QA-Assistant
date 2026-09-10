"""Etkileşimli komut satırı arayüzü (çok turlu sohbet).

Sürekli soru sorma döngüsü çalıştırır; RAG cevabıyla birlikte kaynakları
ve yanıt süresini basar. Önceki turlar hafızada tutulup takip sorularında
kullanılır. /quit yazınca çıkılır, /reset sohbet geçmişini temizler.

Çalıştırma:
    python app_cli.py            # normal mod
    python app_cli.py --debug    # cevapla birlikte kullanılan chunk'ları da gösterir
"""

import argparse

from src.rag import answer_query

EXIT_COMMAND = "/quit"
RESET_COMMAND = "/reset"


def _print_debug_chunks(chunks: list) -> None:
    print("\n--- Kullanılan chunk'lar ---")
    for chunk in chunks:
        print(f"[{chunk.source_file} #{chunk.chunk_index}] score={chunk.score:.4f}")
        print(f"  {chunk.content[:200]}")
    print("---------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(description="MC_RAG interaktif soru-cevap")
    parser.add_argument(
        "--debug", action="store_true", help="Cevapla birlikte kullanılan chunk'ları da göster"
    )
    args = parser.parse_args()

    print(
        f"MC_RAG - offline RAG asistanı. Çıkmak için {EXIT_COMMAND}, "
        f"sohbeti sıfırlamak için {RESET_COMMAND} yazın.\n"
    )

    history: list[dict] = []

    while True:
        try:
            question = input("Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor.")
            break

        if question == EXIT_COMMAND:
            print("Çıkılıyor.")
            break

        if question == RESET_COMMAND:
            history = []
            print("Sohbet geçmişi sıfırlandı.\n")
            continue

        if not question:
            continue

        result = answer_query(question, history=history)

        if args.debug:
            _print_debug_chunks(result["chunks"])

        sources = ", ".join(result["sources"]) if result["sources"] else "-"
        print(f"\nCevap: {result['answer']}")
        print(f"Kaynaklar: {sources}")
        print(f"({result['latency']:.2f} sn)\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()