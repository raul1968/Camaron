import sys
import os
import csv
import json
import re
import shutil
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QListWidget,
    QSplitter, QGroupBox, QScrollArea, QFrame, QComboBox, QLineEdit,
    QCheckBox, QProgressBar, QMessageBox
)
from PyQt6.QtGui import QPixmap, QImage, QPainter, QFont, QFontDatabase
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
import cv2
import pytesseract
from Brain.self_model import AppSelfModel
from Brain.nova_dialogue import NovaDialogueEngine
from Brain.ocr_correction_capsule import OcrCorrectionCapsule
from Brain.review_label_capsule import ReviewLabelCapsule
from Brain.teaching_engine import TeachingEngine
from Brain.user_profile import UserProfileEngine
from Brain.pattern_discovery import PatternDiscoveryEngine
from Brain.memory_consolidation import MemoryConsolidationEngine
from Brain.sentiment_intent import SentimentIntentAnalyzer
from Brain.error_handler import ErrorHandler, handle_error, safe_execute
from Brain.learning_engine import LearningEngine
from Brain.clarification_manager import ClarificationManager
from Brain.autonomous_trainer import AutonomousTrainingScheduler
try:
    from Brain.nova_gru import NovaGRUTrainer
    _GRU_AVAILABLE = True
except ImportError:
    _GRU_AVAILABLE = False
try:
    from Brain.train_storytelling import run_training as run_storytelling_training
    _STORYTELLING_AVAILABLE = True
except ImportError:
    _STORYTELLING_AVAILABLE = False
try:
    from PyQt6.QtMultimedia import QSoundEffect
except ImportError:
    QSoundEffect = None

import numpy as np

try:
    import fitz  # PyMuPDF for PDF rendering and text extraction
    _PDF_AVAILABLE = True
except ImportError:
    fitz = None
    _PDF_AVAILABLE = False

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'}
PDF_EXTENSIONS = {'.pdf'}
TEXT_EXTENSIONS = {'.txt', '.md', '.rtf', '.log'}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS | TEXT_EXTENSIONS

VGG_FEATURES_FILE = os.path.join(REPO_ROOT, 'data', 'vgg_features.h5')
TRAINING_PAIRS_FILE = os.path.join(REPO_ROOT, 'data', 'training_pairs.csv')
PHASE1_TEXT_FILE = os.path.join(REPO_ROOT, 'data', 'phase1_text_training.csv')
AUDIT_FILE = os.path.join(REPO_ROOT, 'data', 'training_pairs_audit.jsonl')
SYSTEM_CONTRACT_FILE = os.path.join(REPO_ROOT, 'Json', 'system_contract.json')
NOVA_TEXT_MODEL_FILE = os.path.join(REPO_ROOT, 'Json', 'nova_text_model.json')
OCR_CAPSULE_FILE = os.path.join(REPO_ROOT, 'Json', 'ocr_correction_capsule.json')
LABEL_CAPSULE_FILE = os.path.join(REPO_ROOT, 'Json', 'review_label_capsule.json')
USER_PROFILE_FILE = os.path.join(REPO_ROOT, 'Json', 'user_profile.json')
MEMORY_CONSOLIDATION_FILE = os.path.join(REPO_ROOT, 'Json', 'memory_consolidation.json')
LEARNING_ENGINE_FILE = os.path.join(REPO_ROOT, 'Json', 'learning_engine.json')
CLARIFICATION_FILE = os.path.join(REPO_ROOT, 'Json', 'clarification_sessions.json')
DRAFTS_FILE = os.path.join(REPO_ROOT, 'Json', 'panel_reviewer_drafts.json')
GRU_CHECKPOINT_FILE = os.path.join(REPO_ROOT, 'Json', 'nova_gru_model.pt')
RESOURCES_DIR = os.path.join(REPO_ROOT, 'resources')
FONT_FILE = os.path.join(RESOURCES_DIR, 'fonts', 'weblysleekuil.ttf')
SOUND_KEY_FILE = os.path.join(RESOURCES_DIR, 'sound', 'key1.ogg')
SOUND_SAVE_FILE = os.path.join(RESOURCES_DIR, 'sound', 'typewriter-bell.ogg')
RESEARCH_PAPERS_FILE = os.path.join(REPO_ROOT, 'Json', 'research_papers.json')

# ── PaperIngestor (Brain module) ──────────────────────────────────────────────
try:
    from Brain.paper_ingestor import PaperIngestor, PaperRecord, load_paper_from_text
    _PAPER_INGESTOR_AVAILABLE = True
except ImportError:
    _PAPER_INGESTOR_AVAILABLE = False
    PaperIngestor = None
    PaperRecord = None
    load_paper_from_text = None

try:
    from Brain.capsule_pipeline import CapsulePipeline
    _CAPSULE_PIPELINE_AVAILABLE = True
except ImportError:
    _CAPSULE_PIPELINE_AVAILABLE = False
    CapsulePipeline = None

CAPSULES_FILE = os.path.join(REPO_ROOT, 'Json', 'capsules.json')


def resolve_tesseract_cmd(contract_path):
    default_windows_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    contract_value = None
    if os.path.exists(contract_path):
        try:
            with open(contract_path, 'r', encoding='utf-8') as f:
                contract = json.load(f)
            contract_value = contract.get('ocr', {}).get('tesseract_cmd')
        except (OSError, json.JSONDecodeError):
            contract_value = None

    env_value = os.environ.get('TESSERACT_CMD') or os.environ.get('TESSERACT_PATH')
    which_value = shutil.which('tesseract')

    for candidate in (contract_value, env_value, which_value, default_windows_path):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


TESSERACT_CMD = resolve_tesseract_cmd(SYSTEM_CONTRACT_FILE)
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Placeholder for network analysis (replace with actual model later)
class PanelAnalyzer:
    def __init__(self):
        self.h5_file = None
        self.features = None
        try:
            import h5py
            self.h5_file = h5py.File(VGG_FEATURES_FILE, 'r')
            if 'features' in self.h5_file:
                self.features = self.h5_file['features']
            elif 'train' in self.h5_file and 'vgg_features' in self.h5_file['train']:
                self.features = self.h5_file['train']['vgg_features']
            elif 'dev' in self.h5_file and 'vgg_features' in self.h5_file['dev']:
                self.features = self.h5_file['dev']['vgg_features']
            elif 'test' in self.h5_file and 'vgg_features' in self.h5_file['test']:
                self.features = self.h5_file['test']['vgg_features']
        except Exception as e:
            print(f"Could not load VGG features: {e}")

    def analyze(self, image_path):
        # Dummy analysis: detect if panel has text or characters
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Simple heuristics
        has_text = pytesseract.image_to_string(gray).strip() != ""
        has_faces = len(cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)) > 0
        
        # Network-like analysis using VGG features if available
        network_insight = "No network data"
        if self.features is not None:
            # Lazy read one feature vector instead of loading entire HDF5 dataset.
            feature = self.features[0] if len(self.features) > 0 else np.zeros(4096)
            network_insight = f"Feature magnitude: {np.linalg.norm(feature):.2f}"
        
        return f"Detected: {'Text' if has_text else 'No Text'}, {'Faces' if has_faces else 'No Faces'}. {network_insight}"

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

class OCRWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit(10, 'Loading panel image for OCR...')
            img = cv2.imread(self.image_path)
            if img is None:
                self.finished.emit('OCR Error: Could not load panel image.')
                return

            self.progress.emit(35, 'Converting panel image to grayscale...')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.progress.emit(75, 'Running Tesseract OCR locally...')
            text = pytesseract.image_to_string(gray)
            self.progress.emit(100, 'OCR complete.')
            self.finished.emit(text)
        except Exception as e:
            self.finished.emit(f"OCR Error: {str(e)}")


class DocumentTextWorker(QThread):
    """Background worker for extracting text from PDFs and text files."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)

    def __init__(self, file_path, file_type, page_num=0):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.page_num = page_num

    def run(self):
        try:
            if self.file_type == 'pdf':
                self._extract_pdf_text()
            elif self.file_type == 'text':
                self._read_text_file()
            else:
                self.finished.emit('OCR Error: Unsupported file type for text extraction.')
        except Exception as e:
            self.finished.emit(f"OCR Error: {str(e)}")

    def _extract_pdf_text(self):
        if not _PDF_AVAILABLE:
            self.finished.emit('OCR Error: PDF support requires PyMuPDF. Install with: pip install PyMuPDF')
            return
        self.progress.emit(10, 'Opening PDF document...')
        doc = fitz.open(self.file_path)
        if self.page_num >= len(doc):
            doc.close()
            self.finished.emit(f'OCR Error: PDF page {self.page_num + 1} is out of range.')
            return
        self.progress.emit(40, f'Extracting text from page {self.page_num + 1}...')
        page = doc[self.page_num]
        text = page.get_text()
        if not text.strip():
            # Scanned PDF — fall back to OCR on rendered page image
            self.progress.emit(60, 'No embedded text. Running OCR on rendered page...')
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
            self.progress.emit(80, 'Running Tesseract OCR on rendered page...')
            text = pytesseract.image_to_string(gray)
            if not text.strip():
                text = '(No text could be extracted or OCR\'d from this page.)'
        doc.close()
        self.progress.emit(100, 'PDF text extraction complete.')
        self.finished.emit(text)

    def _read_text_file(self):
        self.progress.emit(10, 'Reading text file...')
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        self.progress.emit(100, 'Text file loaded.')
        self.finished.emit(text)


class ResearchIngestWorker(QThread):
    """Background worker that extracts research paper metadata from PDF/text files."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)

    def __init__(self, file_path, file_type, extracted_text, page_num=0):
        super().__init__()
        self.file_path = file_path
        self.file_type = file_type
        self.extracted_text = extracted_text
        self.page_num = page_num

    def run(self):
        try:
            self.progress.emit(10, 'Parsing document structure...')
            paper_dict = self._extract_metadata()
            self.progress.emit(50, 'Ingesting as research paper...')
            if _PAPER_INGESTOR_AVAILABLE:
                ingestor = PaperIngestor()
                record = PaperRecord.from_dict(paper_dict)
                result = ingestor.ingest_paper(record)
                paper_dict['ingest_status'] = result.get('status', 'unknown')
                paper_dict['paper_id'] = result.get('paper_id', record.id)
            else:
                import hashlib as _hl
                paper_dict['paper_id'] = _hl.md5(
                    (paper_dict.get('title', '') + paper_dict.get('abstract', '')).encode()
                ).hexdigest()[:12]
                paper_dict['ingest_status'] = 'standalone'
            self.progress.emit(90, 'Saving research record...')
            self._save_record(paper_dict)
            self.progress.emit(100, 'Research ingestion complete.')
            self.finished.emit(paper_dict)
        except Exception as e:
            self.finished.emit({'error': str(e)})

    def _extract_metadata(self):
        text = self.extracted_text or ''
        basename = os.path.splitext(os.path.basename(self.file_path))[0]
        title = basename.replace('_', ' ').replace('-', ' ').title()

        # Try to extract title from first non-empty line
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines and len(lines[0]) < 200:
            title = lines[0]

        # Extract year
        year_match = re.search(r'\b(19|20)\d{2}\b', text[:3000])
        year = year_match.group(0) if year_match else 'unknown'

        # Extract authors (look for common patterns)
        authors = []
        author_patterns = [
            re.compile(r'(?:authors?|by)[:\s]+(.+?)(?:\n|$)', re.IGNORECASE),
            re.compile(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)', re.MULTILINE),
        ]
        for pat in author_patterns:
            m = pat.search(text[:3000])
            if m:
                raw = m.group(1)
                authors = [a.strip() for a in re.split(r',|;|\band\b', raw) if a.strip()]
                if authors:
                    break

        # Extract abstract
        abstract = ''
        abstract_match = re.search(
            r'(?:abstract|summary)[:\s]*\n?(.+?)(?:\n\n|introduction|keywords|\Z)',
            text[:5000], re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:2000]
        elif len(text) > 200:
            abstract = text[:2000]
        else:
            abstract = text

        # Extract key concepts (look for keywords section or frequent capitalized phrases)
        key_concepts = []
        kw_match = re.search(r'(?:keywords?|key\s*terms?)[:\s]*(.+?)(?:\n\n|\Z)',
                             text[:5000], re.IGNORECASE | re.DOTALL)
        if kw_match:
            key_concepts = [k.strip() for k in re.split(r'[,;•·]', kw_match.group(1)) if k.strip()]

        # Extract domain guess (from keywords or title)
        domain = 'unknown'
        domain_keywords = {
            'machine learning': ['neural', 'deep learning', 'machine learning', 'transformer', 'attention'],
            'computer vision': ['image', 'visual', 'detection', 'segmentation', 'CNN'],
            'NLP': ['language', 'NLP', 'text', 'linguistic', 'semantic', 'parser'],
            'robotics': ['robot', 'kinematics', 'actuator', 'manipulation'],
            'comics': ['comic', 'panel', 'manga', 'graphic novel', 'sequential art'],
        }
        text_lower = text[:5000].lower()
        best_count = 0
        for d, kws in domain_keywords.items():
            count = sum(1 for kw in kws if kw.lower() in text_lower)
            if count > best_count:
                best_count = count
                domain = d

        return {
            'title': title,
            'authors': authors[:10],
            'abstract': abstract,
            'year': year,
            'domain': domain,
            'key_concepts': key_concepts[:20],
            'source_file': self.file_path,
            'file_type': self.file_type,
            'page_count': self.page_num + 1 if self.file_type == 'pdf' else 1,
            'text_length': len(text),
        }

    def _save_record(self, paper_dict):
        records = []
        if os.path.exists(RESEARCH_PAPERS_FILE):
            try:
                with open(RESEARCH_PAPERS_FILE, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                records = []
        records.append({
            'paper_id': paper_dict.get('paper_id', ''),
            'title': paper_dict.get('title', ''),
            'authors': paper_dict.get('authors', []),
            'year': paper_dict.get('year', ''),
            'domain': paper_dict.get('domain', ''),
            'abstract': paper_dict.get('abstract', '')[:500],
            'key_concepts': paper_dict.get('key_concepts', []),
            'source_file': paper_dict.get('source_file', ''),
            'file_type': paper_dict.get('file_type', ''),
            'ingested_at': datetime.utcnow().isoformat() + 'Z',
        })
        os.makedirs(os.path.dirname(RESEARCH_PAPERS_FILE), exist_ok=True)
        with open(RESEARCH_PAPERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)


class NovaTrainWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, nova_engine, csv_path):
        super().__init__()
        self.nova_engine = nova_engine
        self.csv_path = csv_path

    def run(self):
        try:
            self.progress.emit(5, 'Opening Phase 1 dataset...')
            self.nova_engine.model = {
                'version': 'v.02',
                'sample_count': 0,
                'intent_emotion': {},
                'intent_only': {},
                'speaker_examples': {},
                'fallback': [],
            }

            if not os.path.exists(self.csv_path):
                self.progress.emit(100, 'No Phase 1 dataset found. Training skipped.')
                self.nova_engine.save()
                self.finished.emit(0)
                return

            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                total_rows = sum(1 for _ in csv.DictReader(f))

            if total_rows == 0:
                self.progress.emit(100, 'Phase 1 dataset is empty. Training skipped.')
                self.nova_engine.save()
                self.finished.emit(0)
                return

            rows = 0
            self.progress.emit(10, f'Training Nova from {total_rows} row(s)...')
            with open(self.csv_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.nova_engine.add_sample(
                        corrected_ocr=row.get('corrected_ocr', ''),
                        speaker=row.get('speaker', 'Unknown'),
                        emotion=row.get('emotion', 'neutral'),
                        intent=row.get('intent', 'statement'),
                    )
                    rows += 1
                    percent = 10 + int((rows / total_rows) * 80)
                    self.progress.emit(percent, f'Training Nova: {rows} of {total_rows} row(s) processed...')

            self.progress.emit(95, 'Saving updated Nova model...')
            self.nova_engine.save()
            self.progress.emit(100, 'Nova training complete.')
            self.finished.emit(rows)
        except Exception as e:
            self.failed.emit(str(e))


class GRUTrainWorker(QThread):
    """Background worker that trains the character-level GRU model."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, gru_trainer, csv_path, epochs=40):
        super().__init__()
        self.gru_trainer = gru_trainer
        self.csv_path = csv_path
        self.epochs = epochs

    def _progress_callback(self, pct, msg):
        self.progress.emit(pct, msg)

    def run(self):
        try:
            self.progress.emit(2, 'Starting Nova GRU training...')
            samples = self.gru_trainer.train_from_csv(
                self.csv_path,
                epochs=self.epochs,
                progress_callback=self._progress_callback,
            )
            self.finished.emit(samples)
        except Exception as e:
            self.failed.emit(str(e))

class StorytellingTrainWorker(QThread):
    """Background worker that trains the multi-task StorytellingBrain model."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, data_dir, folds_dir, output_dir, task='all', epochs=5, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.folds_dir = folds_dir
        self.output_dir = output_dir
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size

    def _progress_callback(self, pct, msg):
        self.progress.emit(pct, msg)

    def run(self):
        try:
            self.progress.emit(1, 'Starting storytelling training...')
            result = run_storytelling_training(
                data_dir=self.data_dir,
                folds_dir=self.folds_dir,
                output_dir=self.output_dir,
                task=self.task,
                epochs=self.epochs,
                batch_size=self.batch_size,
                progress_callback=self._progress_callback,
            )
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class PanelReviewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 1200, 800)

        self.panel_list = []
        self.current_panel = None
        self.boot_warnings = []
        self.analyzer = PanelAnalyzer()
        self.self_model = AppSelfModel(SYSTEM_CONTRACT_FILE)
        if self.self_model.contract_warning:
            self.boot_warnings.append(self.self_model.contract_warning)
        self.ocr_capsule = OcrCorrectionCapsule(OCR_CAPSULE_FILE)
        self.ocr_capsule.hydrate_from_csv(PHASE1_TEXT_FILE)
        if self.ocr_capsule.last_warning:
            self.boot_warnings.append(self.ocr_capsule.last_warning)
        self.label_capsule = ReviewLabelCapsule(LABEL_CAPSULE_FILE)
        self.label_capsule.hydrate_from_csv(PHASE1_TEXT_FILE)
        if self.label_capsule.last_warning:
            self.boot_warnings.append(self.label_capsule.last_warning)
        # Capsule pipeline — bridges review saves to the orbital capsule system
        if _CAPSULE_PIPELINE_AVAILABLE:
            self.capsule_pipeline = CapsulePipeline(store_path=CAPSULES_FILE)
        else:
            self.capsule_pipeline = None
        self.nova_engine = NovaDialogueEngine(NOVA_TEXT_MODEL_FILE)
        self.nova_model_loaded = self.nova_engine.load()
        if self.nova_engine.last_load_warning:
            self.boot_warnings.append(self.nova_engine.last_load_warning)
        self.nova_startup_recap = self.nova_engine.review_startup_recap(AUDIT_FILE)
        self.teaching_engine = TeachingEngine(struggle_threshold=3, cooldown_s=90.0)
        self.user_profile_engine = UserProfileEngine(USER_PROFILE_FILE)
        self.pattern_engine = PatternDiscoveryEngine(min_pattern_size=3)
        self.pattern_engine.ingest_audit_file(AUDIT_FILE)
        self.memory_engine = MemoryConsolidationEngine(MEMORY_CONSOLIDATION_FILE)
        # ── Tier 2 engines ──
        self.sentiment_analyzer = SentimentIntentAnalyzer()
        self.error_handler = ErrorHandler()
        self.learning_engine = LearningEngine(LEARNING_ENGINE_FILE)
        self.clarification_manager = ClarificationManager(CLARIFICATION_FILE)
        self.nova_review_hint = ''
        self.ocr_capsule_hint = ''
        self.ocr_capsule_confidence_hint = ''
        self.label_capsule_hint = ''
        self.sentiment_intent_hint = ''
        self.current_label_suggestions = {}
        self.auto_applied_label_values = {}
        self.current_label_matched_example = ''
        self.raw_ocr_text = ''
        self.unsaved_changes = False
        self._suspend_dirty_tracking = False
        self._skip_next_panel_advance = False
        self.ocr_active = False
        self.analysis_active = False
        self.save_active = False
        self.nova_training_active = False
        self.ocr_worker = None
        self.nova_train_worker = None
        self.gru_train_worker = None
        self.gru_training_active = False
        self.storytelling_train_worker = None
        self.storytelling_training_active = False
        # Document review state (PDF/text support)
        self.current_file_type = 'image'
        self.current_pdf_page = 0
        self.total_pdf_pages = 0
        self.doc_text_worker = None
        # GRU trainer (optional — needs torch)
        self.gru_trainer = None
        if _GRU_AVAILABLE:
            self.gru_trainer = NovaGRUTrainer(GRU_CHECKPOINT_FILE)
            self.gru_trainer.load()
        self.research_ingest_worker = None
        self.research_ingest_active = False
        self.autonomous_scheduler = AutonomousTrainingScheduler(self)
        self.draft_store = self._load_drafts()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setSingleShot(True)
        self.autosave_timer.setInterval(1200)
        self.autosave_timer.timeout.connect(self._autosave_current_draft)
        self.sound_effects = {}
        self.contract = self._load_contract()
        self.offline_enforced = bool(self.contract.get('constraints', {}).get('offline_only', True))
        self.phase1_text_only = True

        identity = self.contract.get('identity', {})
        app_name = identity.get('app_name', 'ROCA Orbit')
        version = identity.get('version', 'v.02')
        self.setWindowTitle(f"{app_name} {version} - Document Reviewer")

        self._apply_custom_font()
        self._setup_sounds()

        self.init_ui()
        self._wire_dirty_tracking()
        self.update_guidance('')
        self.refresh_self_status()
        self._set_operation_progress(0, 'Ready. Select a panel directory to begin review.')

    def _current_context_text(self):
        return f"{self.ocr_text.toPlainText()}\n{self.analysis_label.text()}"

    def _confidence_text(self, confidence, level, basis):
        if confidence <= 0.0:
            return 'OCR Capsule Confidence: no learned cleanup signal yet.'
        basis_text = 'exact correction memory' if basis == 'exact' else 'token correction memory'
        return f'OCR Capsule Confidence: {level.upper()} ({confidence:.2f}) from {basis_text}.'

    def _apply_hint_style(self, label_widget, level, category='ocr'):
        palette = {
            'high': ('#0f5132', '#d1e7dd', '#badbcc'),
            'medium': ('#664d03', '#fff3cd', '#ffecb5'),
            'low': ('#7a1f1f', '#f8d7da', '#f5c2c7'),
            'none': ('#374151', '#e5e7eb', '#d1d5db'),
        }
        text_color, background, border = palette.get(level, palette['none'])
        label_widget.setStyleSheet(
            f'color: {text_color}; background-color: {background}; '
            f'border: 1px solid {border}; border-radius: 6px; padding: 6px;'
        )

    def _suggestion_level(self, suggestion):
        confidences = []
        for key in ('speaker', 'intent', 'utterance_type', 'continuity_link'):
            if key in suggestion:
                confidences.append(suggestion[key].get('confidence', 0.0))
        if not confidences:
            return 'none'
        best = max(confidences)
        if best >= 0.85:
            return 'high'
        if best >= 0.7:
            return 'medium'
        return 'low'

    def _label_field_display_name(self, field_name):
        names = {
            'speaker': 'Speaker',
            'intent': 'Intent',
            'utterance_type': 'Utterance',
            'continuity_link': 'Continuity',
        }
        return names.get(field_name, field_name)

    def _field_current_value(self, field_name):
        if field_name == 'speaker':
            return self.speaker_input.text().strip()
        if field_name == 'intent':
            return self.intent_input.currentText()
        if field_name == 'utterance_type':
            return self.utterance_type_input.currentText()
        if field_name == 'continuity_link':
            return self.continuity_link_input.text().strip()
        return ''

    def _set_field_value(self, field_name, value):
        if field_name == 'speaker':
            self.speaker_input.setText(value)
        elif field_name == 'intent':
            self.intent_input.setCurrentText(value)
        elif field_name == 'utterance_type':
            self.utterance_type_input.setCurrentText(value)
        elif field_name == 'continuity_link':
            self.continuity_link_input.setText(value)

    def _default_field_value(self, field_name):
        defaults = {
            'speaker': '',
            'intent': 'statement',
            'utterance_type': 'dialogue',
            'continuity_link': '',
        }
        return defaults.get(field_name, '')

    def _clear_label_suggestion_controls(self, reason_text=None):
        self.current_label_suggestions = {}
        self.auto_applied_label_values = {}
        self.current_label_matched_example = ''

    def _render_label_suggestion_controls(self, suggestion, applied_fields):
        self.current_label_suggestions = {
            key: value for key, value in suggestion.items()
            if key in ('speaker', 'intent', 'utterance_type', 'continuity_link')
        }
        self.current_label_matched_example = suggestion.get('matched_example', self.current_label_matched_example)



    def _boot_summary(self):
        if not self.boot_warnings:
            return ''
        summary = '; '.join(self.boot_warnings[:3])
        if len(self.boot_warnings) > 3:
            summary += f" (+{len(self.boot_warnings) - 3} more)"
        return f"Startup warnings: {summary}"

    def _set_operation_status(self, message, timeout_ms=0):
        self.operation_status_label.setText(message)
        self.statusBar().showMessage(message, timeout_ms)

    def _set_operation_progress(self, value, message=None, busy=False):
        if busy:
            self.operation_progress_bar.setRange(0, 0)
        else:
            self.operation_progress_bar.setRange(0, 100)
            self.operation_progress_bar.setValue(max(0, min(100, int(value))))

        if message is not None:
            self._set_operation_status(message)

    def _finish_operation_progress(self, message, timeout_ms=0):
        self.operation_progress_bar.setRange(0, 100)
        self.operation_progress_bar.setValue(100)
        self._set_operation_status(message, timeout_ms)

    def _refresh_operation_controls(self):
        busy = self.ocr_active or self.analysis_active or self.save_active
        self.ocr_button.setEnabled((not busy) and bool(self.current_panel))
        self.save_button.setEnabled((not busy) and bool(self.current_panel))
        self.accept_next_button.setEnabled((not busy) and bool(self.current_panel) and self._can_accept_current_panel())
        self.analyze_button.setEnabled((not self.phase1_text_only) and (not busy) and bool(self.current_panel) and self.current_file_type == 'image')
        self.ingest_research_button.setEnabled(
            (not busy) and bool(self.current_panel)
            and self.current_file_type in ('pdf', 'text')
            and not self.research_ingest_active
        )

    def _nova_is_active(self):
        return self.self_model.is_personality_available('Nova')

    def _effective_intent(self, context_text):
        explicit_intent = self.intent_input.currentText()
        return self.nova_engine.infer_intent(context_text, explicit_intent)

    def _auto_apply_label_suggestions(self, corrected_text):
        suggestion = self.label_capsule.suggest(corrected_text)
        if not suggestion:
            self.label_capsule_hint = ''
            self.label_suggestion_label.setText('ROCA label defaults: no learned speaker or intent suggestion yet.')
            self._apply_hint_style(self.label_suggestion_label, 'none', category='label')
            self._clear_label_suggestion_controls()
            return

        applied = []
        applied_fields = set()
        self.auto_applied_label_values = {}

        def apply_defaults():
            speaker = suggestion.get('speaker')
            if speaker and speaker['confidence'] >= 0.78 and not self.speaker_input.text().strip():
                self.auto_applied_label_values['speaker'] = self.speaker_input.text()
                self.speaker_input.setText(speaker['value'])
                applied.append(f"speaker -> {speaker['value']}")
                applied_fields.add('speaker')

            intent = suggestion.get('intent')
            if intent and intent['confidence'] >= 0.78 and self.intent_input.currentText() == 'statement':
                self.auto_applied_label_values['intent'] = self.intent_input.currentText()
                self.intent_input.setCurrentText(intent['value'])
                applied.append(f"intent -> {intent['value']}")
                applied_fields.add('intent')

            utterance_type = suggestion.get('utterance_type')
            if utterance_type and utterance_type['confidence'] >= 0.78 and self.utterance_type_input.currentText() == 'dialogue':
                self.auto_applied_label_values['utterance_type'] = self.utterance_type_input.currentText()
                self.utterance_type_input.setCurrentText(utterance_type['value'])
                applied.append(f"utterance -> {utterance_type['value']}")
                applied_fields.add('utterance_type')

            continuity_link = suggestion.get('continuity_link')
            if continuity_link and continuity_link['confidence'] >= 0.84 and not self.continuity_link_input.text().strip():
                self.auto_applied_label_values['continuity_link'] = self.continuity_link_input.text()
                self.continuity_link_input.setText(continuity_link['value'])
                applied.append(f"continuity -> {continuity_link['value']}")
                applied_fields.add('continuity_link')

        self._with_suspended_dirty_tracking(apply_defaults)

        parts = []
        speaker = suggestion.get('speaker')
        if speaker:
            parts.append(f"speaker {speaker['value']} ({speaker['confidence']:.2f})")
        intent = suggestion.get('intent')
        if intent:
            parts.append(f"intent {intent['value']} ({intent['confidence']:.2f})")
        utterance_type = suggestion.get('utterance_type')
        if utterance_type:
            parts.append(f"utterance {utterance_type['value']} ({utterance_type['confidence']:.2f})")
        continuity_link = suggestion.get('continuity_link')
        if continuity_link:
            parts.append(f"continuity {continuity_link['value']} ({continuity_link['confidence']:.2f})")
        matched_example = suggestion.get('matched_example')
        if matched_example:
            parts.append(f"matched '{matched_example[:60]}'")

        detail = '; '.join(applied if applied else parts)
        prefix = 'ROCA label defaults applied: ' if applied else 'ROCA label defaults suggest: '
        self.label_capsule_hint = prefix + detail
        self.label_suggestion_label.setText(self.label_capsule_hint)
        self._apply_hint_style(self.label_suggestion_label, self._suggestion_level(suggestion), category='label')
        self._render_label_suggestion_controls(suggestion, applied_fields)
        self.memory_engine.record_activation('label', 'label_suggest', {'auto_applied': list(applied_fields)})

    def _can_accept_current_panel(self):
        if not self.current_panel:
            return False
        if not (self.raw_ocr_text or '').strip():
            return False
        corrected_text = self.ocr_text.toPlainText().strip()
        if not corrected_text:
            return False
        if corrected_text.startswith('OCR Error:'):
            return False
        return True

    def _current_ocr_matches_raw(self):
        if not self._can_accept_current_panel():
            return False
        return self.ocr_text.toPlainText().strip() == (self.raw_ocr_text or '').strip()

    def _missing_label_notes(self):
        notes = []
        speaker = self.speaker_input.text().strip()
        continuity = self.continuity_link_input.text().strip()
        if not speaker or speaker.lower() == 'unknown':
            notes.append('speaker is still Unknown')
        if not continuity:
            notes.append('continuity link is blank')
        if self.utterance_type_input.currentText().strip().lower() == 'other':
            notes.append("utterance type is still 'other'")
        if self.intent_input.currentText().strip().lower() == 'other':
            notes.append("intent is still 'other'")
        return notes

    def _prompt_accept_and_continue(self):
        panel_name = os.path.basename(self.current_panel) if self.current_panel else 'current panel'
        message = f'Accept the current OCR text for {panel_name} and continue to the next panel?'
        missing_notes = self._missing_label_notes()
        if missing_notes:
            message += '\n\nLabels still look minimal: ' + '; '.join(missing_notes) + '.'

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle('Accept OCR And Continue')
        box.setText(message)
        accept_button = box.addButton('Accept and Next', QMessageBox.ButtonRole.AcceptRole)
        review_button = box.addButton('Review First', QMessageBox.ButtonRole.RejectRole)
        cancel_button = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(accept_button)
        box.exec()

        clicked = box.clickedButton()
        if clicked == accept_button:
            return 'accept'
        if clicked == review_button:
            return 'review'
        if clicked == cancel_button:
            return 'cancel'
        return 'cancel'

    def _refresh_nova_controls(self):
        if self.nova_training_active:
            self.train_nova_button.setEnabled(False)
            return
        if self.gru_training_active:
            self.train_gru_button.setEnabled(False)
        if self.storytelling_training_active:
            self.train_storytelling_button.setEnabled(False)

        nova_active = self._nova_is_active()
        self.train_nova_button.setEnabled(nova_active)
        if _GRU_AVAILABLE and not self.gru_training_active:
            self.train_gru_button.setEnabled(nova_active)
        if _STORYTELLING_AVAILABLE and not self.storytelling_training_active:
            self.train_storytelling_button.setEnabled(True)
        if nova_active:
            self.train_nova_button.setToolTip('Train Nova from the local Phase 1 text dataset.')
        else:
            self.train_nova_button.setToolTip('Nova is DORMANT under current local hardware limits.')

    def _wire_dirty_tracking(self):
        self.ocr_text.textChanged.connect(self._mark_unsaved_changes)
        self.speaker_input.textChanged.connect(self._mark_unsaved_changes)
        self.utterance_type_input.currentTextChanged.connect(self._mark_unsaved_changes)
        self.emotion_input.currentTextChanged.connect(self._mark_unsaved_changes)
        self.intent_input.currentTextChanged.connect(self._mark_unsaved_changes)
        self.continuity_link_input.textChanged.connect(self._mark_unsaved_changes)

    def _mark_unsaved_changes(self):
        if self._suspend_dirty_tracking:
            return
        if not self.current_panel:
            return
        self.unsaved_changes = True
        self._queue_draft_autosave()

    def _set_unsaved_changes(self, value):
        self.unsaved_changes = bool(value)

    def _with_suspended_dirty_tracking(self, func):
        previous_state = self._suspend_dirty_tracking
        self._suspend_dirty_tracking = True
        try:
            func()
        finally:
            self._suspend_dirty_tracking = previous_state

    def _has_active_background_work(self):
        active_ocr = self.ocr_worker is not None and self.ocr_worker.isRunning()
        active_doc = self.doc_text_worker is not None and self.doc_text_worker.isRunning()
        active_research = self.research_ingest_worker is not None and self.research_ingest_worker.isRunning()
        active_train = self.nova_train_worker is not None and self.nova_train_worker.isRunning()
        active_gru = self.gru_train_worker is not None and self.gru_train_worker.isRunning()
        active_st = self.storytelling_train_worker is not None and self.storytelling_train_worker.isRunning()
        return self.ocr_active or self.analysis_active or self.save_active or self.nova_training_active or self.gru_training_active or self.storytelling_training_active or self.research_ingest_active or active_ocr or active_doc or active_research or active_train or active_gru or active_st

    def _prompt_unsaved_changes(self, reason='exit'):
        if not self.unsaved_changes:
            return 'discard'

        panel_name = os.path.basename(self.current_panel) if self.current_panel else 'current panel'
        message = (
            f'Unsaved reviewer edits exist for {panel_name}. Save before {reason}?'
        )
        choice = QMessageBox.question(
            self,
            'Unsaved Reviewer Changes',
            message,
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if choice == QMessageBox.StandardButton.Save:
            return 'save'
        if choice == QMessageBox.StandardButton.Discard:
            return 'discard'
        return 'cancel'

    def _load_drafts(self):
        if not os.path.exists(DRAFTS_FILE):
            return {}
        try:
            with open(DRAFTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            self.boot_warnings.append(f'Panel reviewer drafts could not be loaded: {exc}')
            return {}

    def _save_drafts(self):
        os.makedirs(os.path.dirname(DRAFTS_FILE), exist_ok=True)
        with open(DRAFTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.draft_store, f, indent=2)

    def _current_draft_payload(self):
        return {
            'panel_path': self.current_panel,
            'raw_ocr_text': self.raw_ocr_text,
            'corrected_ocr': self.ocr_text.toPlainText(),
            'speaker': self.speaker_input.text(),
            'utterance_type': self.utterance_type_input.currentText(),
            'emotion': self.emotion_input.currentText(),
            'intent': self.intent_input.currentText(),
            'continuity_link': self.continuity_link_input.text(),
            'analysis': self.analysis_label.text(),
            'file_type': self.current_file_type,
            'page_num': self.current_pdf_page if self.current_file_type == 'pdf' else None,
            'updated_at': datetime.utcnow().isoformat() + 'Z',
        }

    def _queue_draft_autosave(self):
        if not self.current_panel:
            return
        self.autosave_timer.start()

    def _autosave_current_draft(self):
        if not self.current_panel or not self.unsaved_changes:
            return
        self.draft_store[self.current_panel] = self._current_draft_payload()
        self._save_drafts()
        self._set_operation_status('Draft autosaved locally.', 3000)

    def _remove_draft(self, panel_path=None):
        target_path = panel_path or self.current_panel
        if not target_path:
            return
        if target_path in self.draft_store:
            del self.draft_store[target_path]
            self._save_drafts()

    def _reset_panel_editor_state(self):
        def reset_fields():
            self.ocr_text.clear()
            self.clear_text_labels()

        self._with_suspended_dirty_tracking(reset_fields)
        self.raw_ocr_text = ''
        self.current_pdf_page = 0
        self.total_pdf_pages = 0
        if hasattr(self, 'page_nav_widget'):
            self.page_nav_widget.setVisible(False)
        self.ocr_capsule_hint = ''
        self.ocr_capsule_confidence_hint = ''
        self.label_capsule_hint = ''
        self.ocr_confidence_label.setText('OCR Capsule Confidence: no learned cleanup signal yet.')
        self._apply_hint_style(self.ocr_confidence_label, 'none', category='ocr')
        self.label_suggestion_label.setText('ROCA label defaults: no learned speaker or intent suggestion yet.')
        self._apply_hint_style(self.label_suggestion_label, 'none', category='label')
        self._clear_label_suggestion_controls()
        self.nova_review_hint = ''
        if self.phase1_text_only:
            self.analysis_label.setText('Graphical analysis disabled in Phase 1 text-only mode.')
        else:
            self.analysis_label.setText('Analysis results will appear here.')

    def _restore_draft_for_panel(self, panel_path):
        draft = self.draft_store.get(panel_path)
        if not draft:
            return False

        def apply_draft():
            self.ocr_text.setPlainText(draft.get('corrected_ocr', ''))
            self.speaker_input.setText(draft.get('speaker', ''))
            self.utterance_type_input.setCurrentText(draft.get('utterance_type', 'dialogue'))
            self.emotion_input.setCurrentText(draft.get('emotion', 'neutral'))
            self.intent_input.setCurrentText(draft.get('intent', 'statement'))
            self.continuity_link_input.setText(draft.get('continuity_link', ''))

        self._with_suspended_dirty_tracking(apply_draft)
        self.raw_ocr_text = draft.get('raw_ocr_text', '')
        analysis_text = draft.get('analysis', '')
        if analysis_text:
            self.analysis_label.setText(analysis_text)
        self.ocr_capsule_hint = 'Draft restored from local autosave.'
        self._set_unsaved_changes(True)
        self.update_guidance(self.ocr_text.toPlainText())
        self._set_operation_status('Restored unsaved draft for this panel.', 5000)
        return True

    def _resolve_unsaved_before_navigation(self, reason):
        if self._has_active_background_work():
            QMessageBox.information(
                self,
                'Operation In Progress',
                'The reviewer is still processing OCR, analysis, save, or Nova training. Wait for the current operation to finish before switching panels or directories.',
            )
            return False

        choice = self._prompt_unsaved_changes(reason=reason)
        if choice == 'cancel':
            return False
        if choice == 'save':
            return self.save_corrections(advance_to_next=False)
        if choice == 'discard':
            self.autosave_timer.stop()
            self._remove_draft()
            self._set_unsaved_changes(False)
        return True

    def _persist_runtime_state(self):
        try:
            self.nova_engine.save()
        except OSError as exc:
            QMessageBox.warning(self, 'Nova Save Warning', f'Nova model could not be saved during exit: {exc}')

        try:
            self.ocr_capsule.save()
        except OSError as exc:
            QMessageBox.warning(self, 'OCR Capsule Save Warning', f'OCR capsule memory could not be saved during exit: {exc}')

        try:
            self.label_capsule.save()
        except OSError as exc:
            QMessageBox.warning(self, 'Label Capsule Save Warning', f'ROCA label defaults could not be saved during exit: {exc}')

        # Tier 1 persistence
        try:
            self.user_profile_engine.save()
        except OSError:
            pass
        try:
            self.memory_engine.run_consolidation_cycle(self.ocr_capsule, self.label_capsule)
            self.memory_engine.save()
        except OSError:
            pass

        # Tier 2 persistence
        try:
            self.learning_engine.save()
        except OSError:
            pass
        try:
            self.clarification_manager.save()
        except OSError:
            pass

    def _ensure_csv_schema(self, file_path, expected_header):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=expected_header)
                writer.writeheader()
            return

        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []

        if existing_header == expected_header:
            return

        migrated_rows = []
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    migrated_rows.append({column: row.get(column, '') for column in expected_header})
        except Exception as exc:
            raise OSError(f'Could not migrate CSV schema for {os.path.basename(file_path)}: {exc}') from exc

        backup_path = f'{file_path}.bak'
        shutil.copyfile(file_path, backup_path)
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=expected_header)
            writer.writeheader()
            writer.writerows(migrated_rows)

    def _append_csv_row(self, file_path, header, row_dict):
        self._ensure_csv_schema(file_path, header)
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow({column: row_dict.get(column, '') for column in header})

    def _apply_custom_font(self):
        if not os.path.exists(FONT_FILE):
            return
        font_id = QFontDatabase.addApplicationFont(FONT_FILE)
        if font_id == -1:
            return
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            return
        self.setFont(QFont(families[0], 10))

    def _setup_sounds(self):
        if QSoundEffect is None:
            self.boot_warnings.append('Qt sound support is unavailable; reviewer audio cues are disabled.')
            return

        key_effect = QSoundEffect(self)
        if os.path.exists(SOUND_KEY_FILE):
            key_effect.setSource(QUrl.fromLocalFile(SOUND_KEY_FILE))
            key_effect.setVolume(0.15)
            self.sound_effects['key'] = key_effect

        save_effect = QSoundEffect(self)
        if os.path.exists(SOUND_SAVE_FILE):
            save_effect.setSource(QUrl.fromLocalFile(SOUND_SAVE_FILE))
            save_effect.setVolume(0.3)
            self.sound_effects['save'] = save_effect

    def _play_sound(self, key):
        effect = self.sound_effects.get(key)
        if effect is not None:
            effect.play()

    def _load_contract(self):
        if not os.path.exists(SYSTEM_CONTRACT_FILE):
            self.boot_warnings.append('System contract file is missing; reviewer is using defaults from the self-model.')
            return {}
        try:
            with open(SYSTEM_CONTRACT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.boot_warnings.append(f'System contract could not be loaded in panel reviewer: {exc}')
            return {}

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel: Controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Directory selection
        self.dir_button = QPushButton("Select Directory")
        self.dir_button.clicked.connect(self.select_directory)
        left_layout.addWidget(self.dir_button)

        # Panel list
        self.panel_list_widget = QListWidget()
        self.panel_list_widget.itemClicked.connect(self.load_panel)
        left_layout.addWidget(QLabel("Panels:"))
        left_layout.addWidget(self.panel_list_widget)

        # Buttons
        self.ocr_button = QPushButton("Read Panel (OCR)")
        self.ocr_button.clicked.connect(self.run_ocr)
        left_layout.addWidget(self.ocr_button)

        # Phase mode switch: default to text-only training
        self.phase1_mode_checkbox = QCheckBox("Phase 1 Text-Only Mode")
        self.phase1_mode_checkbox.setChecked(True)
        self.phase1_mode_checkbox.stateChanged.connect(self.toggle_phase_mode)
        left_layout.addWidget(self.phase1_mode_checkbox)

        self.analyze_button = QPushButton("Analyze Panel Graphically")
        self.analyze_button.clicked.connect(self.analyze_panel)
        left_layout.addWidget(self.analyze_button)

        # Save corrections
        self.save_button = QPushButton("Save Corrections")
        self.save_button.clicked.connect(self.save_corrections)
        left_layout.addWidget(self.save_button)

        self.accept_next_button = QPushButton("Accept OCR + Next")
        self.accept_next_button.clicked.connect(self.accept_ocr_and_next)
        left_layout.addWidget(self.accept_next_button)

        self.train_nova_button = QPushButton("Force Train Nova Text")
        self.train_nova_button.setToolTip('Manual override — Nova auto-trains when enough data accumulates.')
        self.train_nova_button.clicked.connect(self.train_nova_text_model)
        left_layout.addWidget(self.train_nova_button)

        self.train_gru_button = QPushButton("Force Train Nova GRU")
        self.train_gru_button.clicked.connect(self.train_nova_gru_model)
        self.train_gru_button.setToolTip('Manual override — Nova auto-trains GRU when enough data accumulates.')
        if not _GRU_AVAILABLE:
            self.train_gru_button.setEnabled(False)
            self.train_gru_button.setToolTip('PyTorch not available — GRU training disabled.')
        left_layout.addWidget(self.train_gru_button)

        self.train_storytelling_button = QPushButton("Force Train Storytelling")
        self.train_storytelling_button.clicked.connect(self.train_storytelling_model)
        self.train_storytelling_button.setToolTip('Manual override — Nova auto-trains StorytellingBrain when enough data accumulates.')
        if not _STORYTELLING_AVAILABLE:
            self.train_storytelling_button.setEnabled(False)
            self.train_storytelling_button.setToolTip('PyTorch not available — storytelling training disabled.')
        left_layout.addWidget(self.train_storytelling_button)

        self.export_brain_button = QPushButton("Export Nova Brain")
        self.export_brain_button.clicked.connect(self._export_brain)
        self.export_brain_button.setToolTip(
            'Package all of Nova\'s learned state into a portable directory (e.g., thumb drive).'
        )
        left_layout.addWidget(self.export_brain_button)

        self.ingest_research_button = QPushButton("Ingest as Research")
        self.ingest_research_button.clicked.connect(self.ingest_as_research)
        self.ingest_research_button.setToolTip(
            'Extract metadata from the current PDF or text file and save as a research paper record.'
        )
        left_layout.addWidget(self.ingest_research_button)

        self.send_to_storyboard_button = QPushButton("Send to Storyboard \u2192")
        self.send_to_storyboard_button.clicked.connect(self._send_to_storyboard)
        self.send_to_storyboard_button.setToolTip(
            'Convert all reviewed panel drafts into a comic project and open the storyboard editor.'
        )
        left_layout.addWidget(self.send_to_storyboard_button)

        self.next_panel_button = QPushButton("Next Panel")
        self.next_panel_button.clicked.connect(self.next_panel)
        left_layout.addWidget(self.next_panel_button)

        operation_group = QGroupBox("Operation Status")
        operation_layout = QVBoxLayout(operation_group)
        self.operation_status_label = QLabel("Ready.")
        self.operation_status_label.setWordWrap(True)
        operation_layout.addWidget(self.operation_status_label)
        self.operation_progress_bar = QProgressBar()
        self.operation_progress_bar.setRange(0, 100)
        self.operation_progress_bar.setValue(0)
        operation_layout.addWidget(self.operation_progress_bar)
        left_layout.addWidget(operation_group)

        left_group = QGroupBox("Controls")
        left_group.setLayout(left_layout)
        main_layout.addWidget(left_group, 1)

        # Right panel: Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Image display
        self.image_label = QLabel("Select a panel to view")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFrameStyle(QFrame.Shape.Box)
        self.image_label.setMinimumSize(400, 300)
        right_layout.addWidget(self.image_label)

        # PDF page navigation
        page_nav_layout = QHBoxLayout()
        self.prev_page_button = QPushButton("\u25C0 Prev Page")
        self.prev_page_button.clicked.connect(self.prev_pdf_page)
        page_nav_layout.addWidget(self.prev_page_button)
        self.page_indicator_label = QLabel("")
        self.page_indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page_nav_layout.addWidget(self.page_indicator_label)
        self.next_page_button = QPushButton("Next Page \u25B6")
        self.next_page_button.clicked.connect(self.next_pdf_page)
        page_nav_layout.addWidget(self.next_page_button)
        self.page_nav_widget = QWidget()
        self.page_nav_widget.setLayout(page_nav_layout)
        self.page_nav_widget.setVisible(False)
        right_layout.addWidget(self.page_nav_widget)

        # OCR result (editable)
        ocr_group = QGroupBox("OCR Text (Editable)")
        ocr_layout = QVBoxLayout(ocr_group)
        self.ocr_text = QTextEdit()
        self.ocr_text.setPlaceholderText("OCR results will appear here. Edit as needed.")
        ocr_layout.addWidget(self.ocr_text)
        self.ocr_confidence_label = QLabel('OCR Capsule Confidence: no learned cleanup signal yet.')
        self.ocr_confidence_label.setWordWrap(True)
        self._apply_hint_style(self.ocr_confidence_label, 'none', category='ocr')
        ocr_layout.addWidget(self.ocr_confidence_label)
        right_layout.addWidget(ocr_group)

        # Analysis result
        analysis_group = QGroupBox("Network Analysis")
        analysis_layout = QVBoxLayout(analysis_group)
        self.analysis_label = QLabel("Graphical analysis disabled in Phase 1 text-only mode.")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        right_layout.addWidget(analysis_group)

        # Research metadata panel
        research_group = QGroupBox("Research Paper Metadata")
        research_layout = QVBoxLayout(research_group)
        self.research_metadata_label = QLabel(
            'Select a PDF or text file and click "Ingest as Research" to extract paper metadata.'
        )
        self.research_metadata_label.setWordWrap(True)
        research_layout.addWidget(self.research_metadata_label)
        right_layout.addWidget(research_group)

        # Phase 1 text annotations
        text_meta_group = QGroupBox("Phase 1 Text Labels")
        text_meta_layout = QVBoxLayout(text_meta_group)

        text_meta_layout.addWidget(QLabel("Speaker"))
        self.speaker_input = QLineEdit()
        self.speaker_input.setPlaceholderText("e.g. Bruce, Narrator, Unknown")
        text_meta_layout.addWidget(self.speaker_input)

        text_meta_layout.addWidget(QLabel("Utterance Type"))
        self.utterance_type_input = QComboBox()
        self.utterance_type_input.addItems(["dialogue", "narration", "sfx", "thought", "other"])
        text_meta_layout.addWidget(self.utterance_type_input)

        text_meta_layout.addWidget(QLabel("Emotion"))
        self.emotion_input = QComboBox()
        self.emotion_input.addItems(["neutral", "joy", "anger", "fear", "sadness", "surprise", "disgust", "determined"])
        text_meta_layout.addWidget(self.emotion_input)

        text_meta_layout.addWidget(QLabel("Intent"))
        self.intent_input = QComboBox()
        self.intent_input.addItems(["statement", "question", "command", "threat", "promise", "reaction", "joke", "other"])
        text_meta_layout.addWidget(self.intent_input)

        text_meta_layout.addWidget(QLabel("Continuity Link"))
        self.continuity_link_input = QLineEdit()
        self.continuity_link_input.setPlaceholderText("Prior panel id/path or short note")
        text_meta_layout.addWidget(self.continuity_link_input)
        self.label_suggestion_label = QLabel('ROCA label defaults: no learned speaker or intent suggestion yet.')
        self.label_suggestion_label.setWordWrap(True)
        self._apply_hint_style(self.label_suggestion_label, 'none', category='label')
        text_meta_layout.addWidget(self.label_suggestion_label)

        right_layout.addWidget(text_meta_group)

        # Persona guidance panel
        guidance_group = QGroupBox("Nova Guidance")
        guidance_layout = QVBoxLayout(guidance_group)
        self.guidance_label = QLabel("Guidance not available yet.")
        self.guidance_label.setWordWrap(True)
        guidance_layout.addWidget(self.guidance_label)
        right_layout.addWidget(guidance_group)

        # Nova Chat panel
        chat_group = QGroupBox("Chat with Nova")
        chat_layout = QVBoxLayout(chat_group)
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMaximumHeight(200)
        self.chat_history.setPlaceholderText("Chat history will appear here...")
        chat_layout.addWidget(self.chat_history)
        chat_input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask Nova about dialogue, characters, story...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        chat_input_row.addWidget(self.chat_input)
        self.chat_send_button = QPushButton("Send")
        self.chat_send_button.clicked.connect(self.send_chat_message)
        chat_input_row.addWidget(self.chat_send_button)
        chat_layout.addLayout(chat_input_row)
        right_layout.addWidget(chat_group)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_widget)

        right_group = QGroupBox("Results")
        right_group_layout = QVBoxLayout(right_group)
        right_group_layout.setContentsMargins(0, 0, 0, 0)
        right_group_layout.addWidget(right_scroll)
        main_layout.addWidget(right_group, 2)

        self.toggle_phase_mode()
        self._refresh_operation_controls()

    def toggle_phase_mode(self):
        self.phase1_text_only = self.phase1_mode_checkbox.isChecked()
        if self.phase1_text_only:
            self.analysis_label.setText("Graphical analysis disabled in Phase 1 text-only mode.")
        else:
            self.analysis_label.setText("Analysis results will appear here.")
        self._refresh_operation_controls()

    def select_directory(self):
        if not self._resolve_unsaved_before_navigation('changing directories'):
            return

        dir_path = QFileDialog.getExistingDirectory(self, "Select Review Directory")
        if dir_path:
            self._set_operation_progress(10, f"Scanning {dir_path} for reviewable files...")
            self.panel_list = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        self.panel_list.append(os.path.join(root, file))
            self.panel_list_widget.clear()
            for p in self.panel_list:
                ft = self._get_file_type(p)
                prefix = {'image': '[IMG]', 'pdf': '[PDF]', 'text': '[TXT]'}.get(ft, '[???]')
                self.panel_list_widget.addItem(f"{prefix} {os.path.basename(p)}")
            self.current_panel = None
            self.current_file_type = 'image'
            self._reset_panel_editor_state()
            self._set_unsaved_changes(False)
            counts = {}
            for p in self.panel_list:
                ft = self._get_file_type(p)
                counts[ft] = counts.get(ft, 0) + 1
            parts = [f"{v} {k}" for k, v in sorted(counts.items())]
            summary = ', '.join(parts) if parts else 'no supported files'
            self._finish_operation_progress(f"Loaded {len(self.panel_list)} file(s) from {dir_path} ({summary}).", 5000)
            self._refresh_operation_controls()

    def load_panel(self, item):
        index = self.panel_list_widget.row(item)
        self.load_panel_by_index(index)

    def load_panel_by_index(self, index):
        if index < 0 or index >= len(self.panel_list):
            return
        target_panel = self.panel_list[index]
        if self.current_panel and target_panel != self.current_panel:
            if not self._resolve_unsaved_before_navigation('switching panels'):
                return
        self.current_panel = target_panel
        self.current_file_type = self._get_file_type(target_panel)
        self.panel_list_widget.setCurrentRow(index)
        self._reset_panel_editor_state()
        if self.current_file_type == 'image':
            pixmap = QPixmap(self.current_panel)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            self.page_nav_widget.setVisible(False)
            self.ocr_button.setText("Read Panel (OCR)")
        elif self.current_file_type == 'pdf':
            self._load_pdf_page(0)
            self.ocr_button.setText("Extract Page Text")
        elif self.current_file_type == 'text':
            self.image_label.clear()
            self.image_label.setText("Text Document")
            self.page_nav_widget.setVisible(False)
            self.ocr_button.setText("Load File Text")
        self._restore_draft_for_panel(self.current_panel)
        self._finish_operation_progress(f"Loaded file {index + 1} of {len(self.panel_list)}: {os.path.basename(self.current_panel)}", 4000)
        self._refresh_operation_controls()
        self._set_unsaved_changes(False)
        if self.current_panel in self.draft_store:
            self._set_unsaved_changes(True)

    def _get_file_type(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            return 'image'
        if ext in PDF_EXTENSIONS:
            return 'pdf'
        if ext in TEXT_EXTENSIONS:
            return 'text'
        return 'unknown'

    def _load_pdf_page(self, page_num):
        if not _PDF_AVAILABLE:
            self.image_label.setText("PDF support requires PyMuPDF.\nInstall with: pip install PyMuPDF")
            self.page_nav_widget.setVisible(False)
            return
        try:
            doc = fitz.open(self.current_panel)
            self.total_pdf_pages = len(doc)
            if page_num < 0 or page_num >= self.total_pdf_pages:
                doc.close()
                return
            self.current_pdf_page = page_num
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            scaled = pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            doc.close()
            self._refresh_page_controls()
        except Exception as e:
            self.image_label.setText(f"PDF render error: {e}")
            self.page_nav_widget.setVisible(False)

    def _refresh_page_controls(self):
        is_pdf = self.current_file_type == 'pdf' and self.total_pdf_pages > 0
        self.page_nav_widget.setVisible(is_pdf)
        if is_pdf:
            self.page_indicator_label.setText(f"Page {self.current_pdf_page + 1} / {self.total_pdf_pages}")
            self.prev_page_button.setEnabled(self.current_pdf_page > 0)
            self.next_page_button.setEnabled(self.current_pdf_page < self.total_pdf_pages - 1)

    def prev_pdf_page(self):
        if self.current_file_type != 'pdf' or self.current_pdf_page <= 0:
            return
        if self.unsaved_changes:
            if not self._resolve_unsaved_before_navigation('changing pages'):
                return
        self._with_suspended_dirty_tracking(lambda: self.ocr_text.clear())
        self.raw_ocr_text = ''
        self._set_unsaved_changes(False)
        self._load_pdf_page(self.current_pdf_page - 1)
        self._refresh_operation_controls()

    def next_pdf_page(self):
        if self.current_file_type != 'pdf' or self.current_pdf_page >= self.total_pdf_pages - 1:
            return
        if self.unsaved_changes:
            if not self._resolve_unsaved_before_navigation('changing pages'):
                return
        self._with_suspended_dirty_tracking(lambda: self.ocr_text.clear())
        self.raw_ocr_text = ''
        self._set_unsaved_changes(False)
        self._load_pdf_page(self.current_pdf_page + 1)
        self._refresh_operation_controls()

    def next_panel(self):
        if not self.panel_list:
            return
        if self.unsaved_changes and self._current_ocr_matches_raw():
            decision = self._prompt_accept_and_continue()
            if decision == 'accept':
                if not self.save_corrections(advance_to_next=False):
                    return
            else:
                return
        current_index = self.panel_list_widget.currentRow()
        if current_index < 0:
            current_index = 0
        next_index = min(current_index + 1, len(self.panel_list) - 1)
        self.load_panel_by_index(next_index)

    def accept_ocr_and_next(self):
        if not self._can_accept_current_panel():
            self._finish_operation_progress('Run OCR first or review the current OCR text before accepting.', 5000)
            return
        if self._has_active_background_work():
            QMessageBox.information(
                self,
                'Operation In Progress',
                'The reviewer is still processing OCR, analysis, save, or Nova training. Wait for the current operation to finish before accepting and continuing.',
            )
            return
        current_index = self.panel_list_widget.currentRow()
        has_next = current_index >= 0 and current_index < (len(self.panel_list) - 1)
        if not self.save_corrections(advance_to_next=False):
            return
        if has_next:
            self.load_panel_by_index(current_index + 1)

    def clear_text_labels(self):
        def clear_fields():
            self.speaker_input.clear()
            self.utterance_type_input.setCurrentIndex(0)
            self.emotion_input.setCurrentIndex(0)
            self.intent_input.setCurrentIndex(0)
            self.continuity_link_input.clear()

        self._with_suspended_dirty_tracking(clear_fields)

    def run_ocr(self):
        if self.current_panel:
            self.ocr_active = True
            self._refresh_operation_controls()
            basename = os.path.basename(self.current_panel)
            if self.current_file_type == 'image':
                self._set_operation_progress(0, f"Running OCR on {basename}...")
                self.ocr_worker = OCRWorker(self.current_panel)
                self.ocr_worker.progress.connect(self._on_ocr_progress)
                self.ocr_worker.finished.connect(self.display_ocr)
                self.ocr_worker.start()
            elif self.current_file_type == 'pdf':
                self._set_operation_progress(0, f"Extracting text from {basename} page {self.current_pdf_page + 1}...")
                self.doc_text_worker = DocumentTextWorker(self.current_panel, 'pdf', self.current_pdf_page)
                self.doc_text_worker.progress.connect(self._on_ocr_progress)
                self.doc_text_worker.finished.connect(self.display_ocr)
                self.doc_text_worker.start()
            elif self.current_file_type == 'text':
                self._set_operation_progress(0, f"Reading {basename}...")
                self.doc_text_worker = DocumentTextWorker(self.current_panel, 'text')
                self.doc_text_worker.progress.connect(self._on_ocr_progress)
                self.doc_text_worker.finished.connect(self.display_ocr)
                self.doc_text_worker.start()

    def _on_ocr_progress(self, value, message):
        self._set_operation_progress(value, message)

    def display_ocr(self, text):
        self.ocr_active = False
        self.raw_ocr_text = text
        self._play_sound('key')
        self.nova_review_hint = self.nova_engine.observe_ocr_result(text)
        if text.startswith('OCR Error:'):
            self.ocr_capsule_hint = ''
            self.ocr_capsule_confidence_hint = ''
            self.label_capsule_hint = ''
            self._with_suspended_dirty_tracking(lambda: self.ocr_text.setPlainText(text))
            self.ocr_confidence_label.setText('OCR Capsule Confidence: unavailable because OCR failed.')
            self.label_suggestion_label.setText('ROCA label defaults: unavailable because OCR failed.')
            self._apply_hint_style(self.ocr_confidence_label, 'low', category='ocr')
            self._apply_hint_style(self.label_suggestion_label, 'low', category='label')
            self._clear_label_suggestion_controls('Why suggested: unavailable because OCR failed.')
            self._finish_operation_progress('OCR failed. Review the error message in the OCR text box.', 7000)
        else:
            suggestion = self.ocr_capsule.suggest_with_confidence(text)
            suggested_text = suggestion['suggested_text']
            changes = suggestion['changes']
            self._with_suspended_dirty_tracking(lambda: self.ocr_text.setPlainText(suggested_text))
            if changes:
                self.ocr_capsule_hint = (
                    'OCR Capsule: Suggested cleanup based on prior corrected panels: '
                    + '; '.join(changes)
                )
            else:
                self.ocr_capsule_hint = ''
            self.ocr_capsule_confidence_hint = self._confidence_text(
                suggestion['confidence'],
                suggestion['level'],
                suggestion['basis'],
            )
            self.ocr_confidence_label.setText(self.ocr_capsule_confidence_hint)
            self._apply_hint_style(self.ocr_confidence_label, suggestion['level'], category='ocr')
            self._auto_apply_label_suggestions(suggested_text)
            self.memory_engine.record_activation('ocr', 'ocr_suggest', {'changes': len(changes)})
            self.user_profile_engine.record_action('panel_review', parameters={'user_input': suggested_text[:80]})
            # Tier 2: sentiment/intent auto-classification of OCR text
            classification = self.sentiment_analyzer.classify(suggested_text)
            if classification.intent.value != self.intent_input.currentText():
                self.sentiment_intent_hint = f"Sentiment analysis: {classification.sentiment.value}, intent looks like '{classification.intent.value}', tone: {classification.emotional_tone.value}."
            else:
                self.sentiment_intent_hint = ''
            line_count = len([line for line in text.splitlines() if line.strip()])
            self._finish_operation_progress(f"OCR complete. Extracted {line_count} non-empty line(s).", 5000)
        self._set_unsaved_changes(not text.startswith('OCR Error:'))
        if self.unsaved_changes:
            self._queue_draft_autosave()
        self._refresh_operation_controls()
        self.update_guidance(self.ocr_text.toPlainText())

    def analyze_panel(self):
        if self.phase1_text_only:
            self.analysis_label.setText("Graphical analysis disabled in Phase 1 text-only mode.")
            return
        if self.current_panel:
            self.analysis_active = True
            self._refresh_operation_controls()
            self._set_operation_progress(20, f"Analyzing {os.path.basename(self.current_panel)} with local heuristics...")
            result = self.analyzer.analyze(self.current_panel)
            self.analysis_label.setText(result)
            combined_context = f"{self.ocr_text.toPlainText()}\n{result}"
            self.analysis_active = False
            self._finish_operation_progress('Graphical analysis complete.', 5000)
            self._refresh_operation_controls()
            self.update_guidance(combined_context)
            self.refresh_self_status()

    def refresh_self_status(self):
        self._refresh_nova_controls()
        self._refresh_operation_controls()

    def update_guidance(self, context_text=''):
        baseline = self.self_model.compose_guidance(context_text)
        boot_summary = self._boot_summary()
        extra_lines = []
        if not context_text.strip() and self.nova_startup_recap:
            extra_lines.append(self.nova_startup_recap)
        if self.nova_review_hint:
            extra_lines.append(self.nova_review_hint)
        if self.ocr_capsule_hint:
            extra_lines.append(self.ocr_capsule_hint)
        if self.ocr_capsule_confidence_hint:
            extra_lines.append(self.ocr_capsule_confidence_hint)
        if self.label_capsule_hint:
            extra_lines.append(self.label_capsule_hint)

        # Tier 2: sentiment/intent classification hint
        if getattr(self, 'sentiment_intent_hint', ''):
            extra_lines.append(self.sentiment_intent_hint)

        # Tier 1: teaching engine tick + user profile suggestions
        teaching_hint = self.teaching_engine.tick()
        if teaching_hint:
            extra_lines.append(teaching_hint)
        for suggestion in self.user_profile_engine.get_adaptation_suggestions():
            extra_lines.append(suggestion['message'])

        # Autonomous training status
        auto_status = self.autonomous_scheduler.get_status_summary()
        if auto_status:
            extra_lines.append(auto_status)

        if self._nova_is_active():
            if not context_text.strip() and not self.nova_model_loaded:
                learned = self.nova_engine.startup_guidance(
                    has_phase1_data=os.path.exists(PHASE1_TEXT_FILE)
                )
                lines = [baseline]
                if boot_summary:
                    lines.append(boot_summary)
                lines.extend(extra_lines)
                lines.append(learned)
                self.guidance_label.setText("\n".join(line for line in lines if line))
                self.refresh_self_status()
                return

            learned = self.nova_engine.generate_guidance(
                context_text=context_text,
                intent=self._effective_intent(context_text),
                emotion=self.emotion_input.currentText(),
                speaker=self.speaker_input.text().strip() or 'Unknown',
            )
        else:
            learned = (
                'Nova: DORMANT under current local hardware limits. '
                'Text guidance generation is paused until Nova becomes ACTIVE.'
            )
        lines = [baseline]
        if boot_summary and not context_text.strip():
            lines.append(boot_summary)
        lines.extend(extra_lines)
        lines.append(learned)
        self.guidance_label.setText("\n".join(line for line in lines if line))
        self.refresh_self_status()

    def train_nova_text_model(self):
        if not self._nova_is_active():
            self.guidance_label.setText(
                'Nova: Training is blocked because Nova is DORMANT under current local hardware limits.'
            )
            self._finish_operation_progress('Nova training blocked because Nova is DORMANT.', 7000)
            self.refresh_self_status()
            return

        self.nova_training_active = True
        self.train_nova_button.setEnabled(False)
        self.guidance_label.setText("Nova: Training text model in background...")
        self._set_operation_progress(0, 'Training Nova from the local Phase 1 dataset...')
        self.nova_train_worker = NovaTrainWorker(self.nova_engine, PHASE1_TEXT_FILE)
        self.nova_train_worker.progress.connect(self._on_nova_train_progress)
        self.nova_train_worker.finished.connect(self._on_train_nova_finished)
        self.nova_train_worker.failed.connect(self._on_train_nova_failed)
        self.nova_train_worker.start()

    def _on_nova_train_progress(self, value, message):
        self._set_operation_progress(value, message)

    def _on_train_nova_finished(self, rows):
        self.nova_model_loaded = True
        self.nova_training_active = False
        self.autonomous_scheduler.on_training_complete('nova_text', {'rows': rows})
        self.guidance_label.setText(
            f"Nova: Text model trained from {rows} rows and saved to {NOVA_TEXT_MODEL_FILE}."
        )
        self._finish_operation_progress(f'Nova training complete. Processed {rows} row(s).', 7000)
        self.refresh_self_status()

    def _on_train_nova_failed(self, error_message):
        self.nova_training_active = False
        self.guidance_label.setText(f"Nova: Training failed: {error_message}")
        self._finish_operation_progress(f'Nova training failed: {error_message}', 10000)
        self.refresh_self_status()

    # ── GRU generative training ───────────────────────────────────────────

    def train_nova_gru_model(self):
        if not _GRU_AVAILABLE or not self.gru_trainer:
            self.guidance_label.setText('Nova GRU: PyTorch is not available. Cannot train.')
            return
        if not self._nova_is_active():
            self.guidance_label.setText(
                'Nova GRU: Training is blocked because Nova is DORMANT under current local hardware limits.'
            )
            self._finish_operation_progress('GRU training blocked because Nova is DORMANT.', 7000)
            self.refresh_self_status()
            return

        self.gru_training_active = True
        self.train_gru_button.setEnabled(False)
        self.guidance_label.setText("Nova GRU: Training character-level neural model in background...")
        self._set_operation_progress(0, 'Training Nova GRU from the local Phase 1 dataset...')
        self.gru_train_worker = GRUTrainWorker(self.gru_trainer, PHASE1_TEXT_FILE, epochs=40)
        self.gru_train_worker.progress.connect(self._on_nova_train_progress)
        self.gru_train_worker.finished.connect(self._on_gru_train_finished)
        self.gru_train_worker.failed.connect(self._on_gru_train_failed)
        self.gru_train_worker.start()

    def _on_gru_train_finished(self, samples):
        self.gru_training_active = False
        self.autonomous_scheduler.on_training_complete('nova_gru', {'samples': samples})
        status = self.gru_trainer.format_report() if self.gru_trainer else 'done'
        self.guidance_label.setText(
            f"Nova GRU: Neural model trained from {samples} sample(s). {status}"
        )
        # Sync GRU into Nova engine so generate_guidance picks it up
        if self.nova_engine.gru_trainer and self.gru_trainer:
            self.nova_engine.gru_trainer = self.gru_trainer
        self._finish_operation_progress(f'GRU training complete. {samples} sample(s) processed.', 7000)
        self.refresh_self_status()

    def _on_gru_train_failed(self, error_message):
        self.gru_training_active = False
        self.guidance_label.setText(f"Nova GRU: Training failed: {error_message}")
        self._finish_operation_progress(f'GRU training failed: {error_message}', 10000)
        self.refresh_self_status()

    # ── Research Paper Ingestion ───────────────────────────────────────────

    def _export_brain(self):
        """Package Nova's learned state for portable copy (e.g., thumb drive)."""
        dest = QFileDialog.getExistingDirectory(
            self, "Choose export destination", "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not dest:
            return

        dest = os.path.join(dest, 'nova_brain')
        reply = QMessageBox.question(
            self, 'Export Nova Brain',
            f'Export all of Nova\'s learned state to:\n{dest}\n\n'
            'This will copy personality, knowledge, models, and chat history.\n'
            'Raw training data (panels, VGG features) will NOT be copied.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._set_operation_progress(10, 'Exporting Nova brain...')
        try:
            from tools.export_nova_brain import export_brain
            manifest = export_brain(dest, include_code=True, include_data_files=True)
            stats = manifest.get('stats', {})
            msg = (f"Brain exported: {stats.get('total_files', '?')} files, "
                   f"{stats.get('human_size', '?')}")
            self._finish_operation_progress(msg, 10000)
            self._append_chat_message('Nova', f'My brain has been exported to {dest}. '
                                      f'{stats.get("total_files", "?")} files, '
                                      f'{stats.get("human_size", "?")}. '
                                      'You can copy that folder to a thumb drive.')
        except Exception as exc:
            self._finish_operation_progress(f'Export failed: {exc}', 8000)

    def _send_to_storyboard(self):
        """Convert reviewed panel drafts into a ComicProject and open the storyboard editor."""
        from Brain.review_handoff import ReviewHandoff

        handoff = ReviewHandoff()
        drafts = handoff.load_drafts()

        if not drafts:
            QMessageBox.information(
                self, 'No Drafts',
                'No reviewed panel drafts found.\n\n'
                'Review and save corrections on some panels first, '
                'then use this button to send them to the storyboard editor.'
            )
            return

        # If current panel has unsaved edits, autosave the draft first
        if self.current_panel and self.unsaved_changes:
            self._autosave_current_draft()

        reply = QMessageBox.question(
            self, 'Send to Storyboard',
            f'Convert {len(drafts)} reviewed panel(s) into a comic project '
            f'and open the storyboard editor?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        project = handoff.drafts_to_project(title='Imported from Review')
        self._set_operation_progress(80, 'Opening storyboard editor...')

        try:
            from tools.storyboard_window import StoryboardWindow
            self._storyboard_win = StoryboardWindow(project=project)
            self._storyboard_win.show()
            self._finish_operation_progress(
                f'Opened storyboard with {len(project.pages)} page(s) '
                f'from {len(drafts)} draft(s).', 8000
            )
        except Exception as exc:
            self._finish_operation_progress(f'Failed to open storyboard: {exc}', 8000)

    def ingest_as_research(self):
        """Extract metadata from the current PDF or text file and ingest as a research paper."""
        if not self.current_panel:
            self._finish_operation_progress('No file selected. Open a PDF or text file first.', 5000)
            return
        if self.current_file_type not in ('pdf', 'text'):
            self._finish_operation_progress('Research ingestion is for PDF and text files only.', 5000)
            return
        if self.research_ingest_active:
            return
        extracted_text = self.ocr_text.toPlainText().strip()
        if not extracted_text:
            self._finish_operation_progress(
                'Extract text first (use Read/Extract button), then ingest as research.', 5000)
            return

        self.research_ingest_active = True
        self._refresh_operation_controls()
        basename = os.path.basename(self.current_panel)
        self._set_operation_progress(0, f'Ingesting {basename} as research paper...')

        # Get total page count for PDFs
        page_total = 0
        if self.current_file_type == 'pdf' and _PDF_AVAILABLE:
            try:
                doc = fitz.open(self.current_panel)
                page_total = len(doc)
                doc.close()
            except Exception:
                page_total = self.total_pdf_pages

        self.research_ingest_worker = ResearchIngestWorker(
            self.current_panel,
            self.current_file_type,
            extracted_text,
            page_num=page_total if page_total else self.current_pdf_page,
        )
        self.research_ingest_worker.progress.connect(self._on_ocr_progress)
        self.research_ingest_worker.finished.connect(self._display_research_result)
        self.research_ingest_worker.start()

    def _display_research_result(self, result):
        self.research_ingest_active = False
        self._refresh_operation_controls()

        if 'error' in result:
            self.research_metadata_label.setText(f'Research ingestion failed: {result["error"]}')
            self._finish_operation_progress(f'Research ingestion failed: {result["error"]}', 7000)
            return

        parts = [
            f"Title: {result.get('title', 'Unknown')}",
            f"Authors: {', '.join(result.get('authors', [])) or 'Not detected'}",
            f"Year: {result.get('year', 'unknown')}",
            f"Domain: {result.get('domain', 'unknown')}",
        ]
        concepts = result.get('key_concepts', [])
        if concepts:
            parts.append(f"Key Concepts: {', '.join(concepts[:8])}")
        abstract = result.get('abstract', '')
        if abstract:
            parts.append(f"Abstract: {abstract[:300]}{'...' if len(abstract) > 300 else ''}")
        parts.append(f"Paper ID: {result.get('paper_id', 'N/A')}")
        parts.append(f"Status: {result.get('ingest_status', 'saved')}")

        self.research_metadata_label.setText('\n'.join(parts))
        self._play_sound('save')
        self._finish_operation_progress(
            f'Research paper ingested: {result.get("title", "Unknown")[:60]}', 7000)
        self.memory_engine.record_activation('research', 'paper_ingest', {
            'paper_id': result.get('paper_id', ''),
            'domain': result.get('domain', ''),
        })

    # ── Storytelling Brain training ────────────────────────────────────────

    def train_storytelling_model(self):
        if not _STORYTELLING_AVAILABLE:
            self.guidance_label.setText('Storytelling: PyTorch is not available. Cannot train.')
            return
        if self.storytelling_training_active:
            return

        self.storytelling_training_active = True
        self.train_storytelling_button.setEnabled(False)
        self.guidance_label.setText('Storytelling Brain: Training multi-task model in background...')
        self._set_operation_progress(0, 'Preparing storytelling training (this may take a while)...')

        data_dir = os.path.join(REPO_ROOT, 'data')
        folds_dir = os.path.join(REPO_ROOT, 'folds')
        output_dir = os.path.join(REPO_ROOT, 'checkpoints')

        self.storytelling_train_worker = StorytellingTrainWorker(
            data_dir=data_dir,
            folds_dir=folds_dir,
            output_dir=output_dir,
            task='all',
            epochs=5,
            batch_size=16,
        )
        self.storytelling_train_worker.progress.connect(self._on_nova_train_progress)
        self.storytelling_train_worker.finished.connect(self._on_storytelling_train_finished)
        self.storytelling_train_worker.failed.connect(self._on_storytelling_train_failed)
        self.storytelling_train_worker.start()

    def _on_storytelling_train_finished(self, result):
        self.storytelling_training_active = False
        self.autonomous_scheduler.on_training_complete('storytelling', result)
        params = result.get('params', 0)
        capsules = result.get('capsule_count', 0)
        best = result.get('best_metrics', {})
        summary_parts = [f'{t}: acc={m.get("accuracy", 0):.2%}' for t, m in best.items()]
        summary = ', '.join(summary_parts) if summary_parts else 'done'
        self.guidance_label.setText(
            f'Storytelling Brain: {params:,} param model trained. {capsules} capsule(s). {summary}'
        )
        self._finish_operation_progress(f'Storytelling training complete. {summary}', 10000)
        self.refresh_self_status()

    def _on_storytelling_train_failed(self, error_message):
        self.storytelling_training_active = False
        self.guidance_label.setText(f'Storytelling Brain: Training failed: {error_message}')
        self._finish_operation_progress(f'Storytelling training failed: {error_message}', 10000)
        self.refresh_self_status()

    # ── Nova Chat ─────────────────────────────────────────────────────────

    def send_chat_message(self):
        msg = self.chat_input.text().strip()
        if not msg:
            return
        self.chat_input.clear()

        # Show user message
        self.chat_history.append(f"<b>You:</b> {msg}")

        # Gather current panel context
        panel_text = self.ocr_text.toPlainText().strip() if hasattr(self, 'ocr_text') else ""
        speaker = self.speaker_input.text().strip() if hasattr(self, 'speaker_input') else "Unknown"

        # Get Nova's reply
        reply = self.nova_engine.chat_reply(
            user_message=msg,
            panel_context=panel_text,
            speaker=speaker if speaker else None,
        )
        self.chat_history.append(f"<b>{reply.split(':')[0]}:</b>{':'.join(reply.split(':')[1:])}")

        # Scroll to bottom
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def save_corrections(self, advance_to_next=True):
        if self.current_panel:
            self.save_active = True
            self._refresh_operation_controls()
            self._set_operation_progress(10, f"Saving corrections for {os.path.basename(self.current_panel)}...")
            corrected_text = self.ocr_text.toPlainText()
            raw_ocr_text = self.raw_ocr_text or corrected_text
            analysis_result = self.analysis_label.text()
            os.makedirs(os.path.dirname(TRAINING_PAIRS_FILE), exist_ok=True)

            timestamp_utc = datetime.utcnow().isoformat() + 'Z'
            mode = 'phase1_text_only' if self.phase1_text_only else 'full_multimodal'

            speaker = self.speaker_input.text().strip() or 'Unknown'
            utterance_type = self.utterance_type_input.currentText()
            emotion = self.emotion_input.currentText()
            intent = self.intent_input.currentText()
            continuity_link = self.continuity_link_input.text().strip()
            file_type = self.current_file_type
            page_num = self.current_pdf_page if file_type == 'pdf' else ''

            phase1_header = [
                'panel_path', 'corrected_ocr', 'speaker', 'utterance_type', 'emotion',
                'intent', 'continuity_link', 'timestamp_utc', 'offline_enforced', 'mode',
                'raw_ocr', 'file_type', 'page_num'
            ]
            phase1_row = {
                'panel_path': self.current_panel,
                'corrected_ocr': corrected_text,
                'speaker': speaker,
                'utterance_type': utterance_type,
                'emotion': emotion,
                'intent': intent,
                'continuity_link': continuity_link,
                'timestamp_utc': timestamp_utc,
                'offline_enforced': self.offline_enforced,
                'mode': mode,
                'raw_ocr': raw_ocr_text,
                'file_type': file_type,
                'page_num': page_num,
            }

            training_pairs_header = [
                'panel_path', 'corrected_ocr', 'analysis', 'speaker', 'utterance_type',
                'emotion', 'intent', 'continuity_link', 'timestamp_utc', 'offline_enforced',
                'mode', 'raw_ocr', 'file_type', 'page_num'
            ]
            training_pairs_row = {
                'panel_path': self.current_panel,
                'corrected_ocr': corrected_text,
                'analysis': analysis_result,
                'speaker': speaker,
                'utterance_type': utterance_type,
                'emotion': emotion,
                'intent': intent,
                'continuity_link': continuity_link,
                'timestamp_utc': timestamp_utc,
                'offline_enforced': self.offline_enforced,
                'mode': mode,
                'raw_ocr': raw_ocr_text,
                'file_type': file_type,
                'page_num': page_num,
            }

            if self.phase1_text_only:
                self._set_operation_progress(25, 'Writing Phase 1 text labels...')
                self._append_csv_row(PHASE1_TEXT_FILE, phase1_header, phase1_row)

            # Incremental local learning after every corrected text save, in all modes.
            self._set_operation_progress(45, 'Updating Nova with the corrected sample...')
            self.nova_engine.add_sample(
                corrected_ocr=corrected_text,
                speaker=speaker,
                emotion=emotion,
                intent=intent,
            )
            self.ocr_capsule.learn(raw_ocr_text, corrected_text)
            self.label_capsule.learn(
                corrected_text=corrected_text,
                speaker=speaker,
                intent=intent,
                utterance_type=utterance_type,
                continuity_link=continuity_link,
            )
            self.nova_engine.save()

            # Capsule pipeline: create/update orbital capsule from this review
            if self.capsule_pipeline:
                try:
                    cap_id, is_new = self.capsule_pipeline.ingest_review(
                        panel_path=self.current_panel,
                        corrected_text=corrected_text,
                        speaker=speaker,
                        intent=intent,
                        utterance_type=utterance_type,
                        emotion=emotion,
                        file_type=file_type,
                        raw_ocr=raw_ocr_text,
                    )
                    # Let personalities evaluate merge/split after every save
                    merges = self.capsule_pipeline.evaluate_merges()
                    splits = self.capsule_pipeline.evaluate_splits()
                    if merges or splits:
                        print(f'Capsule pipeline: {len(merges)} merges, {len(splits)} splits')
                except Exception as e:
                    print(f'Capsule pipeline error (non-fatal): {e}')
            self.nova_model_loaded = True

            self._set_operation_progress(65, 'Writing general training pairs...')
            self._append_csv_row(TRAINING_PAIRS_FILE, training_pairs_header, training_pairs_row)

            self._set_operation_progress(85, 'Appending audit trail entry...')
            audit_entry = {
                'panel_path': self.current_panel,
                'timestamp_utc': timestamp_utc,
                'offline_enforced': self.offline_enforced,
                'mode': mode,
                'file_type': file_type,
                'page_num': page_num,
                'raw_ocr': raw_ocr_text,
                'ocr_capsule': self.ocr_capsule.summary(),
                'label_capsule': self.label_capsule.summary(),
                'text_labels': {
                    'speaker': speaker,
                    'utterance_type': utterance_type,
                    'emotion': emotion,
                    'intent': intent,
                    'continuity_link': continuity_link,
                },
                'contract': self.contract.get('constraints', {}),
                'self_status': self.self_model.self_status(),
            }
            with open(AUDIT_FILE, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + '\n')

            # ── Tier 1 intelligence: teaching, profile, patterns, consolidation ──
            ocr_changed = raw_ocr_text.strip() != corrected_text.strip()
            self.teaching_engine.observe('save_correction', success=True)
            self.user_profile_engine.record_action(
                'save_correction',
                parameters={'user_input': corrected_text, 'speaker': speaker,
                            'intent': intent, 'ocr_changed': ocr_changed},
                success=True,
            )
            if ocr_changed:
                self.user_profile_engine.record_action('ocr_correction')
            self.pattern_engine.ingest_record(audit_entry)
            self.memory_engine.record_activation('ocr', 'ocr_learn', {'text': corrected_text[:80]})
            self.memory_engine.record_activation('label', 'label_learn', {'speaker': speaker, 'intent': intent})
            # Tier 2: learning engine — learn from the correction
            self.learning_engine.learn_from_correction(
                raw_ocr=raw_ocr_text,
                corrected_text=corrected_text,
                speaker=speaker,
                intent=intent,
                utterance_type=utterance_type,
            )

            self.nova_startup_recap = self.nova_engine.review_startup_recap(AUDIT_FILE)
            self.nova_review_hint = self.nova_engine.observe_review_save(
                speaker=speaker,
                utterance_type=utterance_type,
                intent=intent,
                continuity_link=continuity_link,
                corrected_text=corrected_text,
            )

            # ── Autonomous training: let Nova decide if she should train ──
            auto_actions = self.autonomous_scheduler.on_data_saved()
            for action_msg in auto_actions:
                self.chat_history.append(f"<b>Nova:</b> {action_msg}")
                print(f'[autonomous] {action_msg}')

            if self.phase1_text_only:
                print(f"Saved Phase 1 text labels for {self.current_panel} to {PHASE1_TEXT_FILE}")
            print(f"Saved corrections for {self.current_panel} to {TRAINING_PAIRS_FILE}")
            self._play_sound('save')
            combined_context = f"{corrected_text}\n{analysis_result}"
            self.update_guidance(combined_context)
            self.refresh_self_status()
            self.save_active = False
            self.autosave_timer.stop()
            self._remove_draft(self.current_panel)
            self._finish_operation_progress('Corrections saved locally and audit log updated.', 6000)
            self._refresh_operation_controls()
            self.clear_text_labels()
            self.raw_ocr_text = ''
            self.ocr_capsule_hint = ''
            self._set_unsaved_changes(False)
            if advance_to_next:
                self.next_panel()
            return True
        return False

    def closeEvent(self, event):
        if self._has_active_background_work():
            QMessageBox.information(
                self,
                'Operation In Progress',
                'The reviewer is still processing OCR, analysis, save, or Nova training. Wait for the current operation to finish before closing the app.',
            )
            event.ignore()
            return

        choice = self._prompt_unsaved_changes(reason='exit')
        if choice == 'cancel':
            event.ignore()
            return
        if choice == 'save':
            self.save_corrections(advance_to_next=False)
            if self.save_active:
                event.ignore()
                return

        self._persist_runtime_state()
        self.analyzer.close()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PanelReviewer()
    window.show()
    sys.exit(app.exec())