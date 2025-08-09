from src.services.doc_scrap.pdf_splitter import test_saving_pdf, test_is_table, test_text_block_visual, test_pages_starting_with_subheading, test_ask_gpt_starts_page_with_title
from config import RAW_DATA_DIR
import asyncio
if __name__ == "__main__":
    #scrapp_pdf()
    asyncio.run(test_saving_pdf())
    # test_is_table()
    # test_text_block_visual()
    #test_pages_starting_with_subheading()
    #asyncio.run(test_ask_gpt_starts_page_with_title())

