#!/usr/bin/env python3

"""
Manga Context ETL Pipeline (Local RAG Backend)

This script parses unstructured HTML catalogs (Manga volumes, Character profiles) 
and extracts structured metadata into a normalized SQLite database.

DESIGN NOTE:
    This pipeline uses **Japanese prompts** to match the source text language.
    Aligning the prompt language with the input data significantly reduces hallucination 
    rates and improves extraction fidelity in Japanese-tuned models (e.g., Elyza-JP-8B).
        
    If adapting this pipeline for non-Japanese source material, adjust the prompts and model choice accordingly.

Usage:
    python context_main.py --media-dir ./test/media --char-dir ./test/characters --db-path manga_context.db

"""

import os
import sqlite3
import argparse
import sys
import uuid
from typing import Optional, List, Dict, Generator
from pydantic_settings import BaseSettings


try:
    import openai
    import instructor
    from bs4 import BeautifulSoup
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: Required external libraries are missing.", file=sys.stderr)
    print("Please run: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

# ==========================================
# PYDANTIC MODELS (Japanese Context)
# ==========================================

class GroupAffiliation(BaseModel):
    # Description: "Name of the organization, team, or crew explicitly stated in the text (e.g., 'Straw Hat Pirates', 'Marines')."
    GroupName: str = Field(..., description="テキスト内で明記されている組織、チーム、一味の名前（例：「麦わらの一味」「海軍」）。")
    
    # Description: "Specific role or relationship within that group (e.g., 'Captain', 'Navigator', 'Member'). Return null if not stated."
    Relation: Optional[str] = Field(None, description="その組織における具体的な役職や関係性（例：「船長」「航海士」「加盟」）。明記なき場合はnull。")

class ExtractedCharacterData(BaseModel):
    """
    Structure for extracting character data. 
    Field descriptions are in Japanese to guide the LLM's focus on the source text.
    """
    # Description: "Character name notation used in the text (Japanese)."
    Name: str = Field(..., description="テキスト内で使用されているキャラクターの表記（日本語）。")
    
    # --- Context-Specific Data (Strict extraction) ---
    # Description: "Integer age explicitly stated in text. Do not infer from 'boy'/'old man'. Return null if no number."
    Age: Optional[int] = Field(None, description="テキスト内に明記された年齢（整数）。「少年」「老人」などの記述からは推測せず、数値がない場合は必ずnullを返すこと。")
    
    # Description: "Title, Rank, or Role explicitly stated in text (e.g., 'Pirate Hunter', 'Bounty Hunter')."
    RankOrRole: Optional[str] = Field(None, description="テキスト内で明記された肩書き、称号、役割（例：「海賊狩り」「賞金稼ぎ」）。")
    
    # --- Static Data (Upsert Targets) ---
    # Description: "Birthday (e.g., 'May 5')."
    DateOfBirth: Optional[str] = Field(None, description="誕生日（例：「5月5日」）。")
    
    # Description: "Place of origin (e.g., 'East Blue', 'Foosha Village')."
    Origin: Optional[str] = Field(None, description="出身地（例：「東の海」「フーシャ村」）。")
    
    # Description: "Blood type."
    BloodType: Optional[str] = Field(None, description="血液型。")
    
    # Description: "List of affiliated organizations confirmed in the text."
    Groups: List[GroupAffiliation] = Field(default_factory=list, description="テキスト内で確認できる所属組織リスト。")

class LlmOutput(BaseModel):
    CatalogID: str
    CharacterData: List[ExtractedCharacterData]

# ==========================================
# LLM SETUP & PROMPT
# ==========================================

class AppSettings(BaseSettings):
    openai_api_key: str = "vllm"
    openai_base_url: str = "http://localhost:8000/v1"
    
    debug_llm: bool = False
    max_tokens: int = 4096

# Instantiate it once
settings = AppSettings()

def initialize_client() -> openai.OpenAI:
    print("Initializing instructor-patched OpenAI client (Local vLLM)...")
    
    api_key = os.getenv("OPENAI_API_KEY", "vllm")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    # mode=instructor.Mode.MD_JSON allows for reliable JSON extraction even from chatty models
    return instructor.patch(client, mode=instructor.Mode.MD_JSON)

def get_system_prompt() -> str:
    """
    Constructs the system prompt.
    """

    # Translation of System Instructions:
    # "You are an expert in extracting character information from manga/anime summaries.
    #  Strictly follow these rules:
    #  1. **Extract only facts**: Extract only what is written. Do not infer age/relations.
    #  2. **Strictly use null**: If info is missing, return null.
    #  3. **Maintain original text**: Use exact Japanese spelling for names.
    #  4. **Data Types**: Output Age as an integer (int). Remove units like 'years old'."

    return (
        "あなたは、漫画やアニメのテキストから、登場人物の情報を抽出する専門家です。\n"
        "以下のルールを厳守してください：\n"
        "1. **事実のみを抽出**: テキストに書かれていることだけを抽出してください。年齢や関係性を推測することは禁止です。\n"
        "2. **Nullの徹底**: 情報が明記されていないフィールドは、必ず `null` を返してください。\n"
        "3. **原文保持**: 名前や固有名詞は、テキスト内の表記（日本語）をそのまま使用してください。\n"
        "4. **データ型**: 年齢（Age）は必ず整数（int）で出力してください。「歳」などの単位は削除してください。"
    )

def get_few_shot_examples(mode: str) -> List[Dict[str, str]]:
    """
    Returns example interactions to teach the LLM correct formatting and rules.
    """
    if mode == "character":
        # Example: Character Profile Parsing
        input_text = (
            "ポートガス・D・エース\n"
            "白ひげ海賊団 / 2番隊隊長\n"
            "年齢: 20歳\n"
            "出身: 南の海"
        )
        return [
            {"role": "user", "content": f"Extract profile:\n{input_text}"},
            {"role": "assistant", "content": ExtractedCharacterData(
                Name="ポートガス・D・エース",
                Age=20, 
                RankOrRole="海賊",
                Origin="南の海",
                Groups=[GroupAffiliation(GroupName="白ひげ海賊団", Relation="2番隊隊長")]
            ).model_dump_json()}
        ]
    
    elif mode == "media":
        # Example: Story Context Parsing
        input_text = (
            "マリンフォード頂上戦争編\n"
            "エースの処刑を阻止するため、白ひげが海軍本部に乗り込む。\n"
            "ルフィもインペルダウンから脱出し参戦する。"
        )
        return [
            {"role": "user", "content": f"Extract characters for CatalogID 'MARINEFORD':\n{input_text}"},
            {"role": "assistant", "content": LlmOutput(
                CatalogID="MARINEFORD",
                CharacterData=[
                    ExtractedCharacterData(Name="ポートガス・D・エース", RankOrRole="処刑対象", Groups=[]),
                    ExtractedCharacterData(Name="エドワード・ニューゲート", RankOrRole="白ひげ", Groups=[]),
                    ExtractedCharacterData(Name="モンキー・D・ルフィ", RankOrRole="侵入者", Groups=[])
                ]
            ).model_dump_json()}
        ]
    return []


# ==========================================
# DATABASE & LOGIC
# ==========================================

def _initialize_db(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys = ON')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Define Schema (Media, Characters, Groups)
    cursor.execute('''CREATE TABLE IF NOT EXISTS characters (
            character_id_pk TEXT PRIMARY KEY,
            japanese_full_name TEXT,
            preferred_romaji TEXT,
            initial_affiliation TEXT,
            date_of_birth TEXT,
            origin_place TEXT,
            blood_type TEXT,
            hobbies_skills TEXT,
            notes TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS groups (
            group_name_pk TEXT PRIMARY KEY,
            description TEXT
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS media (
            catalog_id_pk TEXT PRIMARY KEY,
            catalog_prefix TEXT,
            title TEXT,
            title_english TEXT,
            description TEXT,
            description_english TEXT,
            publisher_studio TEXT,
            release_date_str TEXT,
            status TEXT DEFAULT 'Released',
            media_type TEXT
    )''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_media_publisher ON media (publisher_studio)')
    
    # Linker tables:
    cursor.execute('''CREATE TABLE IF NOT EXISTS character_groups (
            character_id_fk TEXT NOT NULL,
            group_name_fk TEXT NOT NULL,
            relation TEXT,
            PRIMARY KEY (character_id_fk, group_name_fk),
            FOREIGN KEY (character_id_fk) REFERENCES characters(character_id_pk) ON DELETE CASCADE,
            FOREIGN KEY (group_name_fk) REFERENCES groups(group_name_pk) ON DELETE CASCADE
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS character_media_links (
            catalog_id_fk TEXT NOT NULL,
            character_id_fk TEXT NOT NULL,
            alias_in_context TEXT,
            age_in_entry INTEGER,
            grade_rank TEXT,
            role_notes TEXT,
            PRIMARY KEY (catalog_id_fk, character_id_fk),
            FOREIGN KEY (catalog_id_fk) REFERENCES media(catalog_id_pk) ON DELETE CASCADE,
            FOREIGN KEY (character_id_fk) REFERENCES characters(character_id_pk) ON DELETE CASCADE
    )''')
    
    return conn, cursor

def get_filename_base(path):
    return os.path.splitext(os.path.basename(path))[0]

def discover_html_files(root_dir: str) -> Generator[str, None, None]:
    """Recursively finds all .html files in directory."""
    if not os.path.exists(root_dir):
        return
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.html'):
                yield os.path.join(root, file)

def get_catalog_id(filepath: str) -> str:
    """
    Generates ID from filename, traversing up if index.html
    e.g. 
    ./char.html -> CHAR
    ./char/index.html -> CHAR
    """
    parent_dir = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    if "index" in filename.lower() and parent_dir:
        return parent_dir.upper()
    return os.path.splitext(filename)[0].upper()

def clean_html_for_llm(html_content: str) -> str:
    """
    Removes clutter to save tokens and reduce noise for the LLM.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'footer', 'nav', 'noscript']):
        element.decompose()
    
    text = soup.get_text(separator='\n')
    # Collapse multiple newlines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)[:15000] # Safety cap


def process_and_insert(media_dir: str, char_dir: str, db_path: str, verbose: bool):
    client = initialize_client()
    conn, cursor = _initialize_db(db_path)
    
    # Cache for UUID resolution
    # Map: Japanese Name -> UUID
    name_to_uuid = {}
    
    # Pre-load existing characters to avoid duplicates/confusion
    cursor.execute("SELECT japanese_full_name, character_id_pk FROM characters")
    for row in cursor.fetchall():
        if row['japanese_full_name']:
            name_to_uuid[row['japanese_full_name']] = row['character_id_pk']

    # ==========================================
    # Phase 1: Index Characters (Character Sheets)
    # ==========================================
    print(f"\n--- Phase 1: Parsing Character Sheets ({char_dir}) ---")
    
    for filepath in discover_html_files(char_dir):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_html = f.read()
            cleaned_text = clean_html_for_llm(raw_html)
            
            if verbose: print(f"\n[DEBUG] Processing Profile: {os.path.basename(filepath)}...")

            # 1. LLM Extraction (Character Mode)
            messages = [{"role": "system", "content": get_system_prompt()}]
            messages.extend(get_few_shot_examples("character"))
            messages.append({"role": "user", "content": f"Extract profile:\n\n{cleaned_text}"})

            profile: ExtractedCharacterData = client.chat.completions.create(
                model="elyza/Llama-3-ELYZA-JP-8B",
                response_model=ExtractedCharacterData,
                messages=messages,
                temperature=0.0 # Strict extraction
            )
            
            if verbose: print(f"[DEBUG] Extracted: {profile.Name}")

            # 2. Resolve UUID
            if profile.Name in name_to_uuid:
                char_uuid = name_to_uuid[profile.Name]
            else:
                char_uuid = str(uuid.uuid4())
                name_to_uuid[profile.Name] = char_uuid
            
            # 3. UPSERT Character Data
            # We use COALESCE to keep existing data if new data is null, 
            # but here we prioritize the Character Sheet data as it's likely more accurate.
            cursor.execute('''
                INSERT INTO characters (character_id_pk, japanese_full_name, date_of_birth, origin_place, blood_type)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(character_id_pk) DO UPDATE SET
                    japanese_full_name = excluded.japanese_full_name,
                    date_of_birth = COALESCE(excluded.date_of_birth, characters.date_of_birth),
                    origin_place = COALESCE(excluded.origin_place, characters.origin_place),
                    blood_type = COALESCE(excluded.blood_type, characters.blood_type)
            ''', (char_uuid, profile.Name, profile.DateOfBirth, profile.Origin, profile.BloodType))
            
            # 4. Link Groups
            for grp in profile.Groups:
                cursor.execute("INSERT OR IGNORE INTO groups (group_name_pk) VALUES (?)", (grp.GroupName,))
                cursor.execute('''
                    INSERT INTO character_groups (character_id_fk, group_name_fk, relation)
                    VALUES (?, ?, ?)
                    ON CONFLICT(character_id_fk, group_name_fk) DO UPDATE SET
                        relation = excluded.relation
                ''', (char_uuid, grp.GroupName, grp.Relation))
                
        except Exception as e:
            print(f"Error processing profile {filepath}: {e}", file=sys.stderr)
    
    conn.commit()

    # ==========================================
    # Phase 2: Extract Context (Media Files)
    # ==========================================
    print(f"\n--- Phase 2: Parsing Media Files ({media_dir}) ---")

    for filepath in discover_html_files(media_dir):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_html = f.read()
            cleaned_text = clean_html_for_llm(raw_html)
            catalog_id = get_catalog_id(filepath)
            
            if verbose: print(f"\n[DEBUG] Processing Story: {catalog_id}...")
            
            # 1. LLM Extraction (Media Mode)
            messages = [{"role": "system", "content": get_system_prompt()}]
            messages.extend(get_few_shot_examples("media"))
            messages.append({"role": "user", "content": f"Extract characters for CatalogID '{catalog_id}':\n\n{cleaned_text}"})

            media_data: LlmOutput = client.chat.completions.create(
                model="elyza/Llama-3-ELYZA-JP-8B",
                response_model=LlmOutput,
                messages=messages,
                temperature=0.0
            )

            # 2. Register Media
            # We treat the generic title as a placeholder until parsed properly or if LLM doesn't return Title (LlmOutput schema here doesn't have Title, relying on CatalogID)
            # context_main.py schema had title in LlmOutput but context_test.py removed it to focus on structure. 
            # We'll use a generic title or derived from ID.
            cursor.execute("INSERT OR IGNORE INTO media (catalog_id_pk, title) VALUES (?, ?)", 
                           (catalog_id, f"Story Event: {catalog_id}"))

            # 3. Link Characters
            characters = media_data.CharacterData or []
            for app in characters:
                if not app.Name: continue
                
                # Loose Match Strategy
                target_uuid = None
                if app.Name in name_to_uuid:
                    target_uuid = name_to_uuid[app.Name]
                else:
                    # Fallback: Substring matching
                    for known_name, uuid_val in name_to_uuid.items():
                        if known_name and (app.Name in known_name or known_name in app.Name):
                             target_uuid = uuid_val
                             break
                
                if target_uuid:
                    if verbose: print(f"    Linked {app.Name} -> {catalog_id}")
                    cursor.execute('''
                        INSERT INTO character_media_links (catalog_id_fk, character_id_fk, role_notes, age_in_entry)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(catalog_id_fk, character_id_fk) DO UPDATE SET
                            role_notes = excluded.role_notes,
                            age_in_entry = excluded.age_in_entry
                    ''', (catalog_id, target_uuid, app.RankOrRole, app.Age))
                else:
                    if verbose: print(f"    [Warn] Unmatched Character: {app.Name}")

        except Exception as e:
            print(f"Error processing media {filepath}: {e}", file=sys.stderr)

    conn.commit()
    conn.close()
    print("Database Population Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manga Context DB (Two-Phase Extraction w/ Local LLM)")
    parser.add_argument('--media-dir', default='./data/media')
    parser.add_argument('--char-dir', default='./data/characters')
    parser.add_argument('--db-path', default='manga_context.db')
    # Renamed to match user's preferred flag
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    args = parser.parse_args()
    
    process_and_insert(args.media_dir, args.char_dir, args.db_path, args.verbose)