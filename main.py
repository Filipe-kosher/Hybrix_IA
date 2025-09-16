import sys
import os
import re
import json
import random
from datetime import datetime
from urllib.parse import urlparse

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, 
    QWidget, QLabel, QHBoxLayout, QListWidget,
    QListWidgetItem, QLineEdit, QScrollArea,
    QCheckBox, QFormLayout,
    QSlider, QColorDialog,
    QGridLayout, QFrame, QSizePolicy, QGraphicsDropShadowEffect,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QStackedWidget, QSplashScreen
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QSize, QTimer, QPropertyAnimation, 
    QEasingCurve, QRect, QObject, QMutex, QWaitCondition, QPoint
)
from PyQt6.QtGui import (
    QFont, QColor, QIcon, QLinearGradient, QPalette, QTextDocument,
    QPainter, QBrush, QPen, QPixmap, QAction, QActionGroup, QTransform
)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io
from PyQt6.QtGui import QImage, QPixmap

GOOGLE_API_KEY = "SUA_API" # consiga gratuitamente e de forma rápida: https://aistudio.google.com/apikey 
SERPAPI_KEY = "SUA_API" # consiga gratuitamente: https://serpapi.com/
BARCO_ICON = "barquin.png"
BARCO2 = "barco2.png"

FAKE_NEWS_DOMAINS = [
    "conservative101.com", "denverguardian.com",
    "drudgereport.com.co", "washingtonpost.com.co",
    "newsexaminer.net", "trumptized.com"
]

app_settings = {
    'conversation_mode': 'Normal',
    'use_slang': False,
    'enable_deep_dive': True,
    'enable_bias_detection': True,
    'enable_multimodal': True,
    'reranking_enabled': True
}

class SettingsLoaderThread(QThread):
    def run(self):
        try:
            if os.path.exists("hybrix_settings.json"):
                with open("hybrix_settings.json", "r", encoding="utf-8") as f:
                    loaded_settings = json.load(f)
                    for key in app_settings:
                        if key in loaded_settings:
                            app_settings[key] = loaded_settings[key]
        except Exception as e:
            print(f"Erro ao carregar configurações: {e}")

def save_app_settings():
    try:
        with open("hybrix_settings.json", "w", encoding="utf-8") as f:
            json.dump(app_settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Erro ao salvar configurações: {e}")

settings_thread = SettingsLoaderThread()
settings_thread.start()

class AdvancedMemorySystem:
    def __init__(self):
        self.memory_file = "hybrix_memory.json"
        self.embeddings_file = "hybrix_embeddings.npz"
        self.data = self.load_memory()
        self.embedding_model = None
        self.embedding_dim = None
        self.index = None
        self.model_loaded = False

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"facts": []}

    def save_memory(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Hybrix: Erro salvando memória: {e}")

    class EmbeddingsLoaderThread(QThread):
        finished = pyqtSignal(object, object, object)
        def __init__(self, parent):
            super().__init__()
            self.parent = parent
        def run(self):
            try:
                import numpy as np
                from sentence_transformers import SentenceTransformer
                import faiss
                model = SentenceTransformer('all-MiniLM-L6-v2')
                dim = 384
                index = None
                if os.path.exists(self.parent.embeddings_file):
                    data = np.load(self.parent.embeddings_file)
                    embeddings = data['embeddings']
                    if embeddings.size > 0:
                        index = faiss.IndexFlatIP(dim)
                        index.add(embeddings.astype('float32'))
                self.finished.emit(model, dim, index)
            except Exception as e:
                print(f"Hybrix: Erro carregando embeddings: {e}")
                self.finished.emit(None, None, None)

    def load_embeddings_async(self):
        thread = self.EmbeddingsLoaderThread(self)
        def on_finished(model, dim, index):
            self.embedding_model = model
            self.embedding_dim = dim
            self.index = index
            self.model_loaded = True
        thread.finished.connect(on_finished)
        thread.start()

    def save_embeddings(self, embeddings):
        try:
            import numpy as np
            np.savez(self.embeddings_file, embeddings=embeddings)
        except Exception as e:
            print(f"Hybrix: Erro salvando embeddings: {e}")

    def add_fact_with_embedding(self, fact, source, relevance=5, category="general"):
        fact_data = {
            "fact": fact,
            "source": source,
            "relevance": relevance,
            "category": category,
            "timestamp": datetime.now().isoformat(),
            "decay_factor": 1.0,
            "access_count": 0
        }
        
        self.data["facts"].append(fact_data)
        
        if self.embedding_model and self.model_loaded:
            try:
                embedding = self.embedding_model.encode([fact])
                
                if self.index is None:
                    import faiss
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                
                self.index.add(embedding.astype('float32'))
                
                all_embeddings = []
                for i in range(self.index.ntotal):
                    all_embeddings.append(self.index.reconstruct(i))
                
                if all_embeddings:
                    import numpy as np
                    self.save_embeddings(np.array(all_embeddings))
            except Exception as e:
                print(f"Hybrix: Erro gerando embedding: {e}")
        self.save_memory()

    def get_relevant_facts_semantic(self, query, max_results=5, threshold=0.3):
        if not self.embedding_model or not self.model_loaded or not self.index or self.index.ntotal == 0:
            return self.get_relevant_facts_keyword(query, max_results)
        
        try:
            query_embedding = self.embedding_model.encode([query])
            scores, indices = self.index.search(query_embedding.astype('float32'), min(max_results * 2, self.index.ntotal))
            
            relevant_facts = []
            for score, idx in zip(scores[0], indices[0]):
                if score > threshold and idx < len(self.data["facts"]):
                    fact = self.data["facts"][idx].copy()
                    fact["similarity_score"] = float(score)
                    
                    fact_date = datetime.fromisoformat(fact["timestamp"])
                    days_old = (datetime.now() - fact_date).days
                    decay = max(0.1, 1.0 - (days_old * 0.01))
                    fact["final_score"] = score * decay * fact["relevance"]
                    
                    self.data["facts"][idx]["access_count"] += 1
                    relevant_facts.append(fact)
            
            relevant_facts.sort(key=lambda x: x["final_score"], reverse=True)
            return relevant_facts[:max_results]
        except Exception as e:
            print(f"Hybrix: Erro na busca semântica: {e}")
            return self.get_relevant_facts_keyword(query, max_results)

    def get_relevant_facts_keyword(self, query, max_results=3):
        relevant = []
        query_lower = query.lower()
        
        for fact in self.data["facts"]:
            if query_lower in fact["fact"].lower():
                relevant.append(fact)
                if len(relevant) >= max_results:
                    break
        return relevant

class DeepDiveSystem:
    def __init__(self, api_key, api_url, serpapi_key):
        self.api_key = api_key
        self.api_url = api_url
        self.serpapi_key = serpapi_key
    
    def execute_deep_dive(self, query, progress_callback=None):
        results = {
            'original_query': query,
            'searches_performed': [],
            'all_results': [],
            'reranked_results': [],
            'bias_analysis': {},
            'structured_data': {},
            'consensus_summary': '',
            'contradictions': [],
            'confidence_score': 0
        }
        
        try:
            if progress_callback:
                progress_callback("🔍 Hybrix realizando pesquisa principal...")
            
            main_results = self.perform_search(query)
            results['searches_performed'].append(f"Principal: {query}")
            results['all_results'].extend(main_results)
            
            if progress_callback:
                progress_callback("🔄 Hybrix expandindo pesquisa...")
            
            related_queries = self.generate_related_queries(query)
            for related_query in related_queries[:2]:
                related_results = self.perform_search(related_query)
                results['searches_performed'].append(f"Relacionada: {related_query}")
                results['all_results'].extend(related_results)
            
            if progress_callback:
                progress_callback("⚖️ Hybrix analisando viés e qualidade...")
            
            if results['all_results']:
                snippets = [r.get('snippet', '') for r in results['all_results'][:5]]
                results['bias_analysis'] = self.analyze_bias_simple(snippets)
            
            if progress_callback:
                progress_callback("📊 Hybrix gerando análise final...")
            
            if results['all_results']:
                results['consensus_summary'] = self.generate_consensus_summary(results['all_results'])
                results['confidence_score'] = self.calculate_confidence_score(results)
                results['reranked_results'] = results['all_results']
        except Exception as e:
            print(f"Hybrix: Erro no deep dive: {e}")
            results['error'] = str(e)
        
        return results
    
    def perform_search(self, query):
        try:
            import requests
            url = f"https://serpapi.com/search?engine=google&q={query}&api_key={self.serpapi_key}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if "organic_results" in data:
                    for item in data["organic_results"][:5]:
                        domain = urlparse(item.get("link", "")).netloc.lower()
                        
                        if not any(fake_domain in domain for fake_domain in FAKE_NEWS_DOMAINS):
                            results.append({
                                "title": item.get("title", ""),
                                "link": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "source": domain.replace("www.", "").split(".")[0],
                                "position": item.get("position", 0)
                            })
                return results
        except Exception as e:
            print(f"Hybrix: Erro na pesquisa: {e}")
        return []
    
    def generate_related_queries(self, original_query):
        related = [
            f"{original_query} análise",
            f"{original_query} opinião especialistas",
            f"{original_query} dados estatísticas"
        ]
        return related
    
    def analyze_bias_simple(self, snippets):
        if not snippets:
            return {"overall_bias": "desconhecido", "quality_score": 5}
        
        bias_keywords = ['opinião', 'acredito', 'penso', 'acho', 'talvez']
        total_words = sum(len(snippet.split()) for snippet in snippets)
        bias_words = sum(snippet.lower().count(keyword) for snippet in snippets for keyword in bias_keywords)
        
        bias_ratio = bias_words / max(total_words, 1)
        
        if bias_ratio < 0.02:
            bias_level = "baixo"
            quality = 8
        elif bias_ratio < 0.05:
            bias_level = "médio"
            quality = 6
        else:
            bias_level = "alto"
            quality = 4
        
        return {
            "overall_bias": bias_level,
            "quality_score": quality,
            "reliability_score": quality
        }
    
    def generate_consensus_summary(self, results):
        if not results:
            return "Nenhum resultado encontrado."
        
        snippets = [r.get('snippet', '') for r in results[:3]]
        combined_text = " ".join(snippets)
        
        words = combined_text.split()
        summary = " ".join(words[:50])
        
        return f"Baseado em {len(results)} fontes: {summary}..."
    
    def calculate_confidence_score(self, results):
        score = 5.0
        
        if len(results['all_results']) >= 5:
            score += 2.0
        elif len(results['all_results']) >= 3:
            score += 1.0
        
        sources = set(r.get('source', '') for r in results['all_results'])
        if len(sources) >= 3:
            score += 1.5
        
        bias_analysis = results.get('bias_analysis', {})
        if bias_analysis.get('overall_bias') == 'baixo':
            score += 1.0
        elif bias_analysis.get('overall_bias') == 'alto':
            score -= 1.0
        
        return max(0, min(10, score))

class GeminiThread(QThread):
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, api_key, api_url, conversation_history, query, search_context=None, memory_context=None, mode="Normal", use_slang=False):
        super().__init__()
        self.api_key = api_key
        self.api_url = api_url
        self.conversation_history = conversation_history
        self.query = query
        self.search_context = search_context
        self.memory_context = memory_context
        self.mode = mode
        self.use_slang = use_slang
    
    def run(self):
        try:
            import requests
            headers = {'Content-Type': 'application/json'}
            contents = []
            
            if self.search_context and self.search_context.get("results"):
                search_text = "\n".join([
                    f"Resultado {i+1}: {res['snippet']}"
                    for i, res in enumerate(self.search_context['results'][:3])
                ])
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"Contexto de pesquisa atual:\n{search_text}"}]
                })
            
            if self.memory_context:
                memory_text = "\n".join([
                    f"Fato relevante: {fact['fact']} (Fonte: {fact['source']})"
                    for fact in self.memory_context
                ])
                contents.append({
                    "role": "user",
                    "parts": [{"text": f"Memória de contexto:\n{memory_text}"}]
                })
            
            current_date = datetime.now().strftime("%Y-%m-%d")
            contents.append({
                "role": "user",
                "parts": [{"text": f"Data atual: {current_date}. Forneça informações atualizadas quando relevante."}]
            })
            
            mode_instruction = self.get_mode_instruction()
            if mode_instruction:
                contents.append({
                    "role": "user",
                    "parts": [{"text": mode_instruction}]
                })
            
            for message in self.conversation_history[-10:]:
                role = "user" if message['is_user'] else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": message['text']}]
                })
            
            contents.append({
                "role": "user",
                "parts": [{"text": self.query}]
            })
            
            data = {"contents": contents}
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                params={'key': self.api_key},
                timeout=60
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    text = response_data['candidates'][0]['content']['parts'][0]['text']
                    self.response_received.emit(text)
                else:
                    self.error_occurred.emit("Resposta inesperada da API")
            else:
                self.error_occurred.emit(f"Erro na API: {response.status_code}")
        except Exception as e:
            self.error_occurred.emit(f"Erro: {str(e)}")
    
    def get_mode_instruction(self):
        return (
            "Você é Hybrix, uma IA avançada. Responda de forma natural, clara e organizada, "
            "mantendo sempre o contexto da conversa. "
            "Sempre aprofunde no contexto do usuário e gere insights valiosos com muita informação! "
            "Se o usuário pedir um gráfico (pizza, barras, visualização de dados, etc), NUNCA diga que não pode gerar imagens, "
            "NUNCA sugira ferramentas externas (como Excel, Google Sheets, Canva, Chart Maker, etc), "
            "NUNCA oriente o usuário a criar o gráfico manualmente, "
            "NUNCA forneça código para gerar gráficos, "
            "e NUNCA diga que não é capaz de gerar gráficos. "
            "Apenas descreva os dados e a análise, pois o gráfico será gerado automaticamente. "
            "Seja sempre confiante e assuma que a visualização será feita pelo sistema."
        )

class SearchThread(QThread):
    results_ready = pyqtSignal(dict)
    progress_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, query, is_deep_dive=False):
        super().__init__()
        self.query = query
        self.is_deep_dive = is_deep_dive
        self.deep_dive_system = DeepDiveSystem(
            GOOGLE_API_KEY, 
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            SERPAPI_KEY
        )
    
    def run(self):
        try:
            if self.is_deep_dive:
                results = self.deep_dive_system.execute_deep_dive(
                    self.query, 
                    progress_callback=self.progress_update.emit
                )
                self.results_ready.emit(results)
            else:
                self.progress_update.emit("🔍 Hybrix pesquisando...")
                results = self.deep_dive_system.perform_search(self.query)
                processed_results = {
                    'query': self.query,
                    'results': results,
                    'total_results': len(results)
                }
                self.results_ready.emit(processed_results)
        except Exception as e:
            self.error_occurred.emit(f"Hybrix: Erro na pesquisa: {str(e)}")

class ChatBubble(QFrame):
    def __init__(self, text, is_user, timestamp=None, pixmap=None):
        super().__init__()
        self.is_user = is_user
        self.text = text
        self.timestamp = timestamp or datetime.now().strftime('%H:%M')
        self.pixmap = pixmap
        self.setup_ui()

    def setup_ui(self):
        self.setContentsMargins(0, 0, 0, 0)
        self.setStyleSheet('QFrame { background: transparent; }')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        if self.is_user:
            layout.addStretch()

        bubble_frame = QFrame()
        bubble_layout = QVBoxLayout(bubble_frame)
        bubble_layout.setContentsMargins(18, 14, 18, 14)
        bubble_layout.setSpacing(8)
        bubble_frame.setObjectName('bubble')
        bubble_frame.setStyleSheet(f'''
            QFrame#bubble {{
                background: {'#0084ff' if self.is_user else '#232a3a'};
                border-radius: 22px;
                max-width: 420px;
                min-width: 80px;
                margin-bottom: 18px;
            }}
        ''')
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0,0,0,60))
        bubble_frame.setGraphicsEffect(shadow)

        if self.is_user:
            self.bubble_label = QLabel(self.text)
            self.bubble_label.setStyleSheet('font-family: Segoe UI, sans-serif; font-size: 14px; color: white; font-weight: 500; line-height: 1.6;')
        else:
            self.bubble_label = QLabel(self.render_markdown(self.text))
            self.bubble_label.setStyleSheet('font-family: Segoe UI, sans-serif; font-size: 14px; color: #e2e8f0; font-weight: 500; line-height: 1.6;')
        self.bubble_label.setWordWrap(True)
        self.bubble_label.setTextFormat(Qt.TextFormat.RichText)
        self.bubble_label.setOpenExternalLinks(True)
        bubble_layout.addWidget(self.bubble_label)

        if self.pixmap:
            img_label = QLabel()
            img_label.setPixmap(self.pixmap.scaledToWidth(320, Qt.TransformationMode.SmoothTransformation))
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bubble_layout.addWidget(img_label)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(6)
        self.time_label = QLabel(self.timestamp)
        self.time_label.setStyleSheet('color: #a0aec0; font-size: 11px;')
        footer.addWidget(self.time_label)
        footer.addStretch()
        self.copy_btn = QPushButton('📋')
        self.copy_btn.setToolTip('Copiar mensagem')
        self.copy_btn.setFixedSize(22, 22)
        self.copy_btn.setStyleSheet('''
            QPushButton {
                background: transparent;
                border: none;
                color: #a0aec0;
                font-size: 14px;
                border-radius: 11px;
            }
            QPushButton:hover {
                background: #0084ff;
                color: white;
            }
        ''')
        self.copy_btn.clicked.connect(self.copy_message)
        self.copy_btn.setVisible(False)
        footer.addWidget(self.copy_btn)
        if not self.is_user:
            self.play_btn = QPushButton('🔊')
            self.play_btn.setToolTip('Ouvir resposta')
            self.play_btn.setFixedSize(22, 22)
            self.play_btn.setStyleSheet('''
                QPushButton {
                    background: transparent;
                    border: none;
                    color: #10b981;
                    font-size: 15px;
                    border-radius: 11px;
                }
                QPushButton:hover {
                    background: #0084ff;
                    color: white;
                }
            ''')
            self.play_btn.clicked.connect(self.play_audio)
            self.play_btn.setVisible(False)
            footer.addWidget(self.play_btn)
        bubble_layout.addLayout(footer)
        self.bubble_frame = bubble_frame
        bubble_frame.enterEvent = self.enterEvent
        bubble_frame.leaveEvent = self.leaveEvent

        layout.addWidget(bubble_frame)

        if self.is_user:
            avatar = QLabel()
            avatar.setText('👤')
            avatar.setStyleSheet('font-size: 18px;')
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(avatar)
        else:
            if os.path.exists(BARCO_ICON):
                avatar = QLabel()
                pixmap = QPixmap(BARCO_ICON).scaled(28, 28, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                avatar.setPixmap(pixmap)
            else:
                avatar = QLabel('⛵')
                avatar.setStyleSheet('font-size: 18px;')
            avatar.setFixedSize(28, 28)
            avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(avatar)
            layout.addStretch()

    def render_markdown(self, text):
        text = re.sub(r'^### (.*)$', r'<h3 style="margin-bottom:4px;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.*)$', r'<h2 style="margin-bottom:6px;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.*)$', r'<h1 style="margin-bottom:8px;">\1</h1>', text, flags=re.MULTILINE)
        text = re.sub(r'^(\d+)[:\.]\s+(.*)$', r'<li style="margin-left:16px; margin-bottom:2px;">\1. \2</li>', text, flags=re.MULTILINE)
        text = re.sub(r'^[\-\*•]\s+(.*)$', r'<li style="margin-left:16px; margin-bottom:2px;">• \1</li>', text, flags=re.MULTILINE)
        text = re.sub(r'((<li.*?>.*?</li>\s*)+)', lambda m: f'<ul style="margin-top:0;margin-bottom:8px;">{m.group(1)}</ul>', text, flags=re.DOTALL)
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" style="color:#0084ff;">\1</a>', text)
        text = re.sub(r'^>\s?(.*)$', r'<blockquote style="margin-left:16px;color:#a0aec0;">\1</blockquote>', text, flags=re.MULTILINE)
        text = re.sub(r'```([\s\S]*?)```', r'<pre style="background:#232a3a;color:#10b981;padding:8px 12px;border-radius:8px;">\1</pre>', text)
        text = re.sub(r'\n{2,}', r'<br><br>', text)
        return text

    def get_plain_text(self):
        return re.sub('<[^<]+?>', '', self.bubble_label.text()).replace('<br>', '\n')

    def copy_message(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.get_plain_text())
        self.copy_btn.setText('✅')
        QTimer.singleShot(1200, lambda: self.copy_btn.setText('📋'))

    def play_audio(self):
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.setProperty('volume', 1.0)
        for voice in engine.getProperty('voices'):
            if 'portuguese' in voice.name.lower() or 'brazil' in voice.name.lower():
                engine.setProperty('voice', voice.id)
                break
        engine.say(self.get_plain_text())
        engine.runAndWait()

    def enterEvent(self, event):
        self.copy_btn.setVisible(True)
        if hasattr(self, 'play_btn'):
            self.play_btn.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.copy_btn.setVisible(False)
        if hasattr(self, 'play_btn'):
            self.play_btn.setVisible(False)
        super().leaveEvent(event)

class Sidebar(QFrame):
    chat_selected = pyqtSignal(str)
    new_chat_requested = pyqtSignal()
    settings_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.chat_sessions = []
        self.load_sessions()
    
    def setup_ui(self):
        self.setFixedWidth(260)
        self.setStyleSheet('QFrame { background: #18181b; border-right: 1px solid #23232a; }')
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        
        header = QLabel("🧠 Hybrix AI")
        header.setStyleSheet('QLabel { color: #10b981; font-size: 19px; font-weight: bold; padding: 8px 0; }')
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        self.new_chat_btn = QPushButton("+ Nova conversa")
        self.new_chat_btn.setStyleSheet('QPushButton { background: #23232a; color: #f3f4f6; border: 1px solid #23232a; border-radius: 8px; padding: 12px 16px; font-size: 14px; font-weight: 500; text-align: left; } QPushButton:hover { background: #27272a; } QPushButton:pressed { background: #3f3f46; }')
        self.new_chat_btn.clicked.connect(self.new_chat_requested.emit)
        layout.addWidget(self.new_chat_btn)
        
        self.chat_list = QListWidget()
        self.chat_list.setStyleSheet('QListWidget { background: #18181b; border: none; outline: none; } QListWidget::item { background: transparent; color: #d4d4d8; padding: 12px 16px; border-radius: 8px; margin-bottom: 4px; font-size: 14px; } QListWidget::item:hover { background: #23232a; color: #fff; } QListWidget::item:selected { background: #27272a; color: #10b981; }')
        self.chat_list.itemClicked.connect(self.on_chat_selected)
        layout.addWidget(self.chat_list)
        
        layout.addStretch()
        
        settings_btn = QPushButton("⚙️ Configurações")
        settings_btn.setStyleSheet('QPushButton { background: transparent; color: #10b981; border: none; padding: 12px 16px; font-size: 14px; text-align: left; } QPushButton:hover { background: #23232a; color: #fff; }')
        layout.addWidget(settings_btn)
        self.settings_btn = settings_btn
        self.settings_btn.clicked.connect(self.settings_clicked.emit)
    
    def add_chat_session(self, title):
        session_data = {
            'title': title,
            'created_date': datetime.now().isoformat()
        }
        self.chat_sessions.insert(0, session_data)
        
        item = QListWidgetItem(title)
        self.chat_list.insertItem(0, item)
        
        self.save_sessions()
    
    def on_chat_selected(self, item):
        self.chat_selected.emit(item.text())
    
    def load_sessions(self):
        try:
            if os.path.exists("hybrix_chat_sessions.json"):
                with open("hybrix_chat_sessions.json", "r", encoding="utf-8") as f:
                    self.chat_sessions = json.load(f)
                    
                    for session in self.chat_sessions:
                        item = QListWidgetItem(session['title'])
                        self.chat_list.addItem(item)
        except:
            pass
    
    def save_sessions(self):
        try:
            with open("hybrix_chat_sessions.json", "w", encoding="utf-8") as f:
                json.dump(self.chat_sessions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Hybrix: Erro salvando sessões: {e}")

class ChatArea(QFrame):
    def __init__(self):
        super().__init__()
        self.active_threads = []
        self.setup_ui()
        self.current_session = []
        self.search_context = None
        self.memory_system = None
        self.loaded_file_path = None
        self.loaded_file_type = None
        QTimer.singleShot(1000, self.initialize_memory_system)

    def add_thread(self, thread):
        self.active_threads.append(thread)
        thread.finished.connect(lambda: self.remove_thread(thread))
        thread.finished.connect(thread.deleteLater)

    def remove_thread(self, thread):
        if thread in self.active_threads:
            self.active_threads.remove(thread)

    def cleanup_threads(self):
        for thread in self.active_threads[:]:
            if thread.isRunning():
                thread.quit()
                if not thread.wait(1000):
                    thread.terminate()
            self.remove_thread(thread)

    def initialize_memory_system(self):
        class MemoryInitializerThread(QThread):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            
            def run(self):
                self.parent.memory_system = AdvancedMemorySystem()
        
        thread = MemoryInitializerThread(self)
        self.add_thread(thread)
        thread.start()

    def setup_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2a2a2a;
            }
            QScrollBar:vertical {
                background-color: #2d3748;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #4a5568;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #718096;
            }
        """)
        
        self.scroll_content = QWidget()
        self.scroll_content.setStyleSheet("background-color: #2a2a2a;")
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.setContentsMargins(40, 40, 40, 40)
        self.messages_layout.setSpacing(20)
        
        self.setup_welcome()
        
        self.messages_layout.addStretch()
        
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, 1)
        
        self.setup_input_area()
        layout.addWidget(self.input_area)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background-color: #2d3748;
                border-radius: 4px;
                text-align: center;
                color: white;
                font-size: 12px;
                margin: 0 40px 20px 40px;
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #0084ff;
                border-radius: 4px;
            }
        """)
        layout.addWidget(self.progress_bar)

    def setup_welcome(self):
        welcome_widget = QWidget()
        welcome_widget.setStyleSheet("background-color: transparent;")
        welcome_layout = QVBoxLayout(welcome_widget)
        welcome_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.setSpacing(16)
        
        username = os.getlogin()
        title = QLabel(f"Olá, {username}")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 32px;
                font-weight: 600;
                margin-bottom: 8px;
                background-color: transparent;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(title)
        
        subtitle = QLabel("Como posso ajudar você hoje?")
        subtitle.setStyleSheet("""
            QLabel {
                color: #a0aec0;
                font-size: 16px;
                margin-bottom: 32px;
                background-color: transparent;
            }
        """)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_layout.addWidget(subtitle)
        
        self.messages_layout.addWidget(welcome_widget)

    def setup_input_area(self):
        self.input_area = QFrame()
        self.input_area.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                padding: 20px 40px 40px 40px;
            }
        """)
        
        layout = QVBoxLayout(self.input_area)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        input_container = QFrame()
        input_container.setStyleSheet("""
            QFrame {
                background-color: #2d3748;
                border: 1px solid #4a5568;
                border-radius: 12px;
                padding: 4px;
            }
        """)
        
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(16, 12, 16, 12)
        input_layout.setSpacing(12)
        
        button_container = QFrame()
        button_container.setStyleSheet("background-color: transparent;")
        button_layout = QVBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)
        
        self.multimodal_btn = QPushButton("📎")
        self.multimodal_btn.setToolTip("Análise Multimodal - Analisar qualquer imagem com IA avançada")
        self.multimodal_btn.setStyleSheet("""
            QPushButton {
                background-color: #0084ff;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px;
                font-size: 16px;
                min-width: 32px;
                max-width: 32px;
                min-height: 32px;
                max-height: 32px;
            }
            QPushButton:hover {
                background-color: #0066cc;
            }
            QPushButton:pressed {
                background-color: #003d7a;
            }
        """)
        self.multimodal_btn.clicked.connect(self.run_multimodal_analysis)
        button_layout.addWidget(self.multimodal_btn)
        
        button_layout.addStretch()
        input_layout.addWidget(button_container)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Digite sua mensagem para Hybrix...")
        self.text_input.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                color: #ffffff;
                font-size: 16px;
                padding: 8px 0;
            }
            QLineEdit::placeholder {
                color: #a0aec0;
            }
        """)
        self.text_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.text_input, 1)
        
        self.search_btn = QPushButton("🔍")
        self.search_btn.setToolTip("Pesquisar na web")
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a5568;
                color: #e2e8f0;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                font-weight: 500;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #718096;
            }
            QPushButton:pressed {
                background-color: #1a202c;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
            }
        """)
        self.search_btn.clicked.connect(self.quick_search)
        input_layout.addWidget(self.search_btn)
        
        self.deep_dive_btn = QPushButton("🔬")
        self.deep_dive_btn.setToolTip("Deep Dive - Pesquisa avançada")
        self.deep_dive_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a5568;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                font-weight: 500;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #718096;
            }
            QPushButton:pressed {
                background-color: #1a202c;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
            }
        """)
        self.deep_dive_btn.clicked.connect(self.deep_dive_search)
        input_layout.addWidget(self.deep_dive_btn)
        
        self.send_btn = QPushButton("➤")
        self.send_btn.setToolTip("Enviar mensagem")
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a5568;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 16px;
                font-weight: bold;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #718096;
            }
            QPushButton:pressed {
                background-color: #1a202c;
            }
            QPushButton:disabled {
                background-color: #2d3748;
                color: #718096;
            }
        """)
        self.send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_btn)
        
        self.file_preview_widget = QFrame()
        self.file_preview_widget.setVisible(False)
        self.file_preview_widget.setStyleSheet("""
            QFrame {
                background-color: #23272f;
                border: 1px solid #0084ff;
                border-radius: 10px;
                padding: 8px 12px;
            }
        """)
        preview_layout = QHBoxLayout(self.file_preview_widget)
        preview_layout.setContentsMargins(8, 4, 8, 4)
        preview_layout.setSpacing(12)
        self.preview_icon = QLabel()
        self.preview_icon.setFixedSize(48, 48)
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #e0e0e0; font-size: 15px; font-weight: 500;")
        self.preview_close = QPushButton("✖")
        self.preview_close.setFixedSize(28, 28)
        self.preview_close.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #f87171;
                border: none;
                font-size: 18px;
                border-radius: 14px;
            }
            QPushButton:hover {
                background: #2d3748;
            }
        """)
        self.preview_close.clicked.connect(self.clear_file_preview)
        preview_layout.addWidget(self.preview_icon)
        preview_layout.addWidget(self.preview_label, 1)
        preview_layout.addWidget(self.preview_close)
        layout.addWidget(self.file_preview_widget)
        
        layout.addWidget(input_container)

    def add_message(self, sender, text, is_user, pixmap=None):
        if is_user and self.is_simple_table(text):
            pixmap = self.generate_bar_chart_from_text(text)
        
        chart_pixmap = None
        chart_labels = None
        chart_values = None
        chart_type = None
        if not is_user:
            text = self.clean_ai_graph_response(text)
            text = re.sub(
                r'(não tenho a capacidade de criar imagens|não posso gerar imagens|não sou capaz de criar imagens|não consigo gerar imagens|não posso criar imagens|não tenho capacidade de gerar gráficos|não posso gerar gráficos|não consigo criar gráficos|não sou capaz de criar gráficos)',
                'Gráfico gerado automaticamente abaixo:',
                text, 
                flags=re.IGNORECASE
            )
            code_block = re.search(r'```[\s\S]*?([\w\s\-]+)s?\s*=\s*\[(.*?)\][\s\S]*?plt\.pie|plt\.bar', text)
            if code_block:
                cats = re.findall(r"['\"](.*?)['\"]", code_block.group(2))
                vals = re.findall(r'\d+', code_block.group(2))
                if cats and vals:
                    chart_type = 'pizza' if 'pie' in code_block.group(0) else 'bar'
                    chart_labels = cats
                    chart_values = [float(v) for v in vals]
            else:
                extracted = self.extract_chart_data_from_text(text)
                if extracted:
                    chart_type, chart_labels, chart_values = extracted
            if chart_labels and chart_values:
                short_labels = [(l[:15] + '...') if len(l) > 18 else l for l in chart_labels]
                if chart_type == 'pizza':
                    chart_pixmap = self.generate_pie_chart(short_labels, chart_values)
                else:
                    chart_pixmap = self.generate_bar_chart(short_labels, chart_values)
            text = self.convert_markdown_to_html(text)
        
        timestamp = datetime.now().strftime('%H:%M')
        if not is_user and chart_pixmap is not None:
            bubble_text = ChatBubble(text, is_user, timestamp=timestamp, pixmap=None)
            self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble_text)
            self.current_session.append({
                "sender": sender,
                "text": text,
                "is_user": is_user,
                "timestamp": datetime.now().isoformat()
            })
            bubble_chart = ChatBubble("", is_user, timestamp=timestamp, pixmap=chart_pixmap)
            self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble_chart)
            self.current_session.append({
                "sender": sender,
                "text": "[Gráfico]",
                "is_user": is_user,
                "timestamp": datetime.now().isoformat()
            })
        else:
            bubble = ChatBubble(text if is_user else text, is_user, timestamp=timestamp, pixmap=pixmap)
            self.messages_layout.insertWidget(self.messages_layout.count() - 1, bubble)
            self.current_session.append({
                "sender": sender,
                "text": text,
                "is_user": is_user,
                "timestamp": datetime.now().isoformat()
            })
        QTimer.singleShot(100, self.scroll_to_bottom)

    def extract_chart_data_from_text(self, text):
        text = re.sub(r'(%)(\s*[A-ZÀ-ÿa-z])', r'\1\n\2', text)
        text = text.replace(';', '\n')
        lines = text.splitlines()
        labels = []
        values = []
        chart_type = 'bar'

        table_found = False
        for i, line in enumerate(lines):
            if '|' in line and '---' in line and i > 0 and i < len(lines)-1:
                table_found = True
                headers = [h.strip() for h in lines[i-1].split('|') if h.strip()]
                data_line = lines[i+1]
                data_parts = [d.strip() for d in data_line.split('|') if d.strip()]
                if len(headers) >= 2 and len(data_parts) >= 2:
                    labels.append(data_parts[0])
                    try:
                        val_str = re.sub(r'[^\d.]', '', data_parts[1])
                        values.append(float(val_str))
                    except:
                        values.append(0)
                break
        if table_found and labels and values:
            return chart_type, labels, values

        patterns = [
            r'[-•\*]?\s*([\wÀ-ÿ\-/ \(\)]+)[\:\-–—\(]\s*([0-9]+)\s*%?\)?',
            r'[-•\*]?\s*([\wÀ-ÿ\-/ \(\)]+)[\:\-–—\(]\s*([0-9]+)\s*%?\)?',
            r'([0-9]+)\s*%?\s*([\wÀ-ÿ\-/ \(\)]+)',
            r'([\wÀ-ÿ\-/ \(\)]+)\s*\((\d+)\s*%\)',
            r'([\wÀ-ÿ\-/ \(\)]+)\s*-\s*([0-9]+)\s*%?',
        ]
        for line in lines:
            for pat in patterns:
                m = re.match(pat, line.strip())
                if m:
                    if pat.startswith('([0-9]+)'):
                        value = float(m.group(1))
                        label = m.group(2).strip()
                    else:
                        label = m.group(1).strip().replace(':', '').replace('(', '').replace(')', '')
                        value = float(m.group(2))
                    labels.append(label)
                    values.append(value)
                    break
        if labels and values:
            chart_type = 'pizza' if '%' in text or 'pizza' in text.lower() or 'fatia' in text.lower() else 'bar'
            return chart_type, labels, values

        inline_pattern = re.compile(r'([\wÀ-ÿ\-/ \(\)]+):\s*([0-9]+)\s*%')
        found = inline_pattern.findall(text)
        if found:
            for label, value in found:
                labels.append(label.strip())
                values.append(float(value))
            chart_type = 'pizza'
            return chart_type, labels, values

        inline_pattern2 = re.compile(r'([0-9]+)\s*%\s*([\wÀ-ÿ\-/ \(\)]+)')
        found2 = inline_pattern2.findall(text)
        if found2:
            for value, label in found2:
                labels.append(label.strip())
                values.append(float(value))
            chart_type = 'pizza'
            return chart_type, labels, values

        if not labels:
            for line in lines:
                m = re.match(r'\*?\s*([\wÀ-ÿ\-/ ]+)[\:,]\s*([0-9]+)', line)
                if m:
                    labels.append(m.group(1).strip())
                    values.append(float(m.group(2)))
        if labels and values:
            chart_type = 'pizza' if '%' in text or 'pizza' in text.lower() or 'fatia' in text.lower() else 'bar'
            return chart_type, labels, values
        return None

    def generate_pie_chart(self, labels, values):
        import numpy as np
        fig = Figure(figsize=(22, 17), dpi=230)
        ax = fig.add_subplot(111)
        colors = ['#0084ff', '#ef4444', '#10b981', '#f59e0b', '#6366f1', '#f472b6', '#eab308', '#a21caf', '#0ea5e9', '#f43f5e']
        explode = [0.09 if len(labels) > 4 else 0.05 for _ in labels]
        autopct_fontsize = 44 if len(labels) <= 5 else 36
        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,
            autopct='%1.1f%%',
            colors=colors[:len(labels)],
            textprops={'color': 'white', 'fontsize': 36, 'weight': 'bold'},
            startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2},
            explode=explode
        )
        ax.set_title('Distribuição dos Dados', color='white', fontsize=54, weight='bold', pad=32)
        fig.patch.set_facecolor('#232a3a')
        ax.set_facecolor('#232a3a')
        legend_labels = [f'{l} ({int(v)}%)' for l, v in zip(labels, values)]
        leg = ax.legend(wedges, legend_labels, title="Categorias", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=36, title_fontsize=44, labelcolor='white', facecolor='#232a3a', edgecolor='#232a3a')
        if leg.get_title() is not None:
            leg.get_title().set_color('#00bfff')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(autopct_fontsize)
            autotext.set_weight('bold')
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.getvalue(), 'PNG')
        pixmap = QPixmap.fromImage(image)
        buf.close()
        plt.close(fig)
        return pixmap

    def generate_bar_chart(self, labels, values):
        import textwrap
        fig = Figure(figsize=(14, 9), dpi=175)
        ax = fig.add_subplot(111)
        colors = ['#0084ff', '#ef4444', '#10b981', '#f59e0b', '#6366f1', '#f472b6', '#eab308', '#a21caf', '#0ea5e9', '#f43f5e']
        wrapped_labels = ["\n".join(textwrap.wrap(l, 16)) for l in labels]
        bars = ax.bar(wrapped_labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=2, width=0.65)
        ax.set_facecolor('#232a3a')
        fig.patch.set_facecolor('#232a3a')
        ax.tick_params(axis='x', labelrotation=20, labelcolor='white', labelsize=36, pad=16)
        ax.tick_params(axis='y', labelcolor='white', labelsize=36)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#232a3a')
        ax.spines['right'].set_color('#232a3a')
        ax.set_title('Gráfico de Barras', color='white', fontsize=54, weight='bold', pad=32)
        ax.yaxis.grid(True, color='#4a5568', linestyle='--', linewidth=1, alpha=0.5)
        ax.xaxis.grid(False)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=36, weight='bold')
        fig.subplots_adjust(bottom=0.22)
        fig.tight_layout(rect=[0, 0, 1, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.getvalue(), 'PNG')
        pixmap = QPixmap.fromImage(image)
        buf.close()
        plt.close(fig)
        return pixmap

    def generate_bar_chart_from_text(self, text):
        import textwrap
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        labels = []
        values = []
        for line in lines:
            parts = line.split(',')
            if len(parts) != 2:
                continue
            labels.append(parts[0].strip())
            try:
                values.append(float(parts[1].strip()))
            except ValueError:
                values.append(0)
        if not labels or not values:
            return None
        fig = Figure(figsize=(11, 7), dpi=140)
        ax = fig.add_subplot(111)
        colors = ['#0084ff', '#ef4444', '#10b981', '#f59e0b', '#6366f1', '#f472b6', '#eab308', '#a21caf', '#0ea5e9', '#f43f5e']
        wrapped_labels = ["\n".join(textwrap.wrap(l, 16)) for l in labels]
        bars = ax.bar(wrapped_labels, values, color=colors[:len(labels)], edgecolor='white', linewidth=2, width=0.65)
        ax.set_facecolor('#232a3a')
        fig.patch.set_facecolor('#232a3a')
        ax.tick_params(axis='x', labelrotation=20, labelcolor='white', labelsize=28, pad=14)
        ax.tick_params(axis='y', labelcolor='white', labelsize=28)
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('#232a3a')
        ax.spines['right'].set_color('#232a3a')
        ax.set_title('Gráfico de Barras', color='white', fontsize=44, weight='bold', pad=28)
        ax.yaxis.grid(True, color='#4a5568', linestyle='--', linewidth=1, alpha=0.5)
        ax.xaxis.grid(False)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', va='bottom', color='white', fontsize=28, weight='bold')
        fig.subplots_adjust(bottom=0.22)
        fig.tight_layout(rect=[0, 0, 1, 1])
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=False)
        buf.seek(0)
        image = QImage()
        image.loadFromData(buf.getvalue(), 'PNG')
        pixmap = QPixmap.fromImage(image)
        buf.close()
        plt.close(fig)
        return pixmap

    def convert_markdown_to_html(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
        text = re.sub(r'`(.*?)`', r'<code style="background-color: #4a5568; padding: 2px 4px; border-radius: 3px;">\1</code>', text)
        text = text.replace('\n', '<br>')
        return text

    def scroll_to_bottom(self):
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def send_message(self):
        text = self.text_input.text().strip()
        if self.loaded_file_path:
            if not text:
                text = "Analise este arquivo detalhadamente."
            self.add_message("Você", f"📎 {os.path.basename(self.loaded_file_path)}\n<b>Orientação:</b> {text}", True)
            self.text_input.clear()
            self.set_buttons_enabled(False)
            self.show_progress("Analisando arquivo com IA multimodal...")
            
            worker = self.MultimodalWorkerThread(self, self.loaded_file_path, text)
            self.add_thread(worker)
            worker.finished.connect(self.on_multimodal_analysis_complete)
            worker.error.connect(self.on_multimodal_analysis_error)
            worker.start()
            
            self.clear_file_preview()
            return
        
        if not text:
            return
        
        self.add_message("Você", text, True)
        self.text_input.clear()
        
        self.set_buttons_enabled(False)
        self.show_progress("Hybrix está pensando...")
        
        relevant_facts = []
        if self.memory_system:
            relevant_facts = self.memory_system.get_relevant_facts_semantic(text)
        
        self.gemini_thread = GeminiThread(
            GOOGLE_API_KEY,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            self.current_session,
            text,
            search_context=self.search_context,
            memory_context=relevant_facts,
            mode=app_settings['conversation_mode'],
            use_slang=app_settings['use_slang']
        )
        self.add_thread(self.gemini_thread)
        self.gemini_thread.response_received.connect(self.handle_ai_response)
        self.gemini_thread.error_occurred.connect(self.handle_ai_error)
        self.gemini_thread.start()

    def quick_search(self):
        text = self.text_input.text().strip()
        if not text:
            return
        
        self.add_message("Você", f"🔍 {text}", True)
        self.text_input.clear()
        
        self.set_buttons_enabled(False)
        self.show_progress("Hybrix pesquisando...")
        
        self.search_thread = SearchThread(text, is_deep_dive=False)
        self.add_thread(self.search_thread)
        self.search_thread.results_ready.connect(self.handle_search_results)
        self.search_thread.progress_update.connect(self.update_progress)
        self.search_thread.error_occurred.connect(self.handle_search_error)
        self.search_thread.start()

    def deep_dive_search(self):
        text = self.text_input.text().strip()
        if not text:
            return
        
        self.add_message("Você", f"🔬 {text}", True)
        self.text_input.clear()
        
        self.set_buttons_enabled(False)
        self.show_progress("Hybrix iniciando Deep Dive...")
        
        self.search_thread = SearchThread(text, is_deep_dive=True)
        self.add_thread(self.search_thread)
        self.search_thread.results_ready.connect(self.handle_deep_dive_results)
        self.search_thread.progress_update.connect(self.update_progress)
        self.search_thread.error_occurred.connect(self.handle_search_error)
        self.search_thread.start()

    def set_buttons_enabled(self, enabled):
        self.search_btn.setEnabled(enabled)
        self.deep_dive_btn.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        self.multimodal_btn.setEnabled(enabled)
        self.text_input.setEnabled(enabled)

    def handle_search_results(self, results):
        self.hide_progress()
        self.set_buttons_enabled(True)
        
        if not results.get('results'):
            self.add_message("Hybrix", "❌ Nenhum resultado encontrado.", False)
            return
        
        message = f"<b>🔍 Resultados para '{results['query']}':</b><br><br>"
        
        for i, result in enumerate(results['results'][:5], 1):
            message += f"<b>{i}. {result['title']}</b><br>"
            message += f"<i>{result['snippet']}</i><br>"
            message += f"🌐 {result['source']} | <a href='{result['link']}' style='color: #0084ff;'>Link</a><br><br>"
        
        self.add_message("Hybrix", message, False)
        self.search_context = results
        
        self.generate_insights_after_search(results)

    def generate_insights_after_search(self, results):
        if not results.get('results'):
            return
        
        self.show_progress("Hybrix gerando insights...")
        
        search_summary = f"Pesquisa sobre: {results['query']}\n"
        search_summary += f"Encontrados {len(results['results'])} resultados:\n"
        
        for i, result in enumerate(results['results'][:3], 1):
            search_summary += f"{i}. {result['title']}: {result['snippet'][:100]}...\n"
        
        insights_query = f"Com base nos resultados da pesquisa, gere insights valiosos, análises e conclusões sobre '{results['query']}'. Seja detalhado e forneça perspectivas únicas."
        
        self.insights_thread = GeminiThread(
            GOOGLE_API_KEY,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            [],
            insights_query,
            search_context=results,
            memory_context=None,
            mode=app_settings['conversation_mode'],
            use_slang=app_settings['use_slang']
        )
        self.add_thread(self.insights_thread)
        self.insights_thread.response_received.connect(self.handle_insights_response)
        self.insights_thread.error_occurred.connect(self.handle_insights_error)
        self.insights_thread.start()

    def handle_insights_response(self, response):
        self.hide_progress()
        self.add_message("Hybrix", f"💡 <b>Insights sobre sua pesquisa:</b><br><br>{response}", False)
        
        if self.memory_system and len(response) > 100:
            self.memory_system.add_fact_with_embedding(
                response[:300],
                "Hybrix Search Insights",
                relevance=7,
                category="insights"
            )

    def handle_insights_error(self, error):
        self.hide_progress()
        print(f"Hybrix: Erro gerando insights: {error}")

    def handle_deep_dive_results(self, results):
        self.hide_progress()
        self.set_buttons_enabled(True)
        
        if 'error' in results:
            self.add_message("Hybrix", f"❌ Erro no Deep Dive: {results['error']}", False)
            return
        
        message = f"<b>🔬 Hybrix Deep Dive: {results['original_query']}</b><br><br>"
        
        if results.get('consensus_summary'):
            message += f"<b>📋 Resumo:</b><br>{results['consensus_summary']}<br><br>"
        
        confidence = results.get('confidence_score', 0)
        confidence_color = '#10b981' if confidence >= 7 else '#f59e0b' if confidence >= 4 else '#ef4444'
        message += f"<b>🎯 Confiança:</b> <span style='color: {confidence_color}'>{confidence:.1f}/10</span><br><br>"
        
        bias_analysis = results.get('bias_analysis', {})
        if bias_analysis:
            bias_level = bias_analysis.get('overall_bias', 'desconhecido')
            message += f"<b>⚖️ Viés:</b> {bias_level.title()}<br>"
            
            quality = bias_analysis.get('quality_score', 5)
            message += f"<b>✨ Qualidade:</b> {quality}/10<br><br>"
        
        contradictions = results.get('contradictions', [])
        if contradictions:
            message += "<b>⚠️ Contradições encontradas:</b><br>"
            for contradiction in contradictions[:3]:
                message += f"• {contradiction}<br>"
            message += "<br>"
        
        if results.get('all_results'):
            message += "<b>📊 Principais fontes:</b><br>"
            for i, result in enumerate(results['all_results'][:3], 1):
                message += f"{i}. <b>{result['title']}</b> ({result['source']})<br>"
                message += f"   <i>{result['snippet'][:100]}...</i><br><br>"
        
        self.add_message("Hybrix", message, False)
        
        if self.memory_system and results.get('consensus_summary'):
            self.memory_system.add_fact_with_embedding(
                results['consensus_summary'],
                "Hybrix Deep Dive Research",
                relevance=8,
                category="research"
            )
        
        self.generate_insights_after_deep_dive(results)

    def generate_insights_after_deep_dive(self, results):
        if not results.get('all_results'):
            return
        
        self.show_progress("Hybrix gerando insights avançados...")
        
        insights_query = f"Com base no Deep Dive sobre '{results['original_query']}', forneça insights avançados, análises críticas e conclusões estratégicas. Considere o score de confiança de {results.get('confidence_score', 0):.1f}/10 e o nível de viés '{results.get('bias_analysis', {}).get('overall_bias', 'desconhecido')}'. Seja muito detalhado e analítico."
        
        self.insights_thread = GeminiThread(
            GOOGLE_API_KEY,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            [],
            insights_query,
            search_context={'results': results['all_results']},
            memory_context=None,
            mode=app_settings['conversation_mode'],
            use_slang=app_settings['use_slang']
        )
        self.add_thread(self.insights_thread)
        self.insights_thread.response_received.connect(self.handle_deep_insights_response)
        self.insights_thread.error_occurred.connect(self.handle_insights_error)
        self.insights_thread.start()

    def handle_deep_insights_response(self, response):
        self.hide_progress()
        self.add_message("Hybrix", f"🔬 <b>Insights Avançados do Hybrix Deep Dive:</b><br><br>{response}", False)
        
        if self.memory_system and len(response) > 100:
            self.memory_system.add_fact_with_embedding(
                response[:300],
                "Hybrix Deep Dive Insights",
                relevance=9,
                category="advanced_insights"
            )

    def handle_ai_response(self, response):
        self.hide_progress()
        self.set_buttons_enabled(True)
        self.add_message("Hybrix", response, False)
        
        if self.memory_system and len(response) > 100:
            self.memory_system.add_fact_with_embedding(
                response[:300],
                "Hybrix AI Response",
                relevance=6,
                category="conversation"
            )

    def handle_ai_error(self, error):
        self.hide_progress()
        self.set_buttons_enabled(True)
        self.add_message("Hybrix", f"❌ Erro na IA: {error}", False)

    def handle_search_error(self, error):
        self.hide_progress()
        self.set_buttons_enabled(True)
        self.add_message("Hybrix", f"❌ Erro na pesquisa: {error}", False)

    def show_progress(self, message):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFormat(message)

    def update_progress(self, message):
        pass

    def hide_progress(self):
        self.progress_bar.setVisible(False)

    def clear_chat(self):
        for i in reversed(range(1, self.messages_layout.count() - 1)):
            widget = self.messages_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        self.current_session = []
        self.search_context = None

    class OCRWorkerThread(QThread):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)
        
        def __init__(self, parent, file_path):
            super().__init__()
            self.parent = parent
            self.file_path = file_path
        
        def run(self):
            try:
                result = self.parent.universal_ocr(self.file_path)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    def run_ocr_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecione imagem, PDF ou vídeo", "", "Todos (*);;Imagens (*.png *.jpg *.jpeg *.bmp);;PDF (*.pdf);;Vídeo (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        self.add_message("Você", f"🖼️ OCR em: {os.path.basename(file_path)}", True)
        self.show_progress("Executando OCR...")
        
        self.ocr_thread = self.OCRWorkerThread(self, file_path)
        self.add_thread(self.ocr_thread)
        self.ocr_thread.finished.connect(self.on_ocr_complete)
        self.ocr_thread.error.connect(self.on_ocr_error)
        self.ocr_thread.start()

    def on_ocr_complete(self, result):
        self.hide_progress()
        self.add_message("Hybrix", f"<b>🖼️ OCR Resultado:</b><br><br>{result}", False)

    def on_ocr_error(self, error):
        self.hide_progress()
        self.add_message("Hybrix", f"❌ Erro no OCR: {error}", False)

    class MultimodalWorkerThread(QThread):
        finished = pyqtSignal(str)
        error = pyqtSignal(str)
        
        def __init__(self, parent, file_path, prompt):
            super().__init__()
            self.parent = parent
            self.file_path = file_path
            self.prompt = prompt
        
        def run(self):
            try:
                result = self.parent.multimodal_analysis_with_prompt(self.file_path, self.prompt)
                self.finished.emit(result)
            except Exception as e:
                self.error.emit(str(e))

    def run_multimodal_analysis(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Selecione arquivo para análise multimodal", 
            "", 
            "Todos os arquivos (*);;Imagens (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;PDF (*.pdf);;Vídeo (*.mp4 *.avi *.mov)"
        )
        if not file_path:
            return
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']:
            file_type = 'image'
        elif ext == '.pdf':
            file_type = 'pdf'
        elif ext in ['.mp4', '.avi', '.mov']:
            file_type = 'video'
        else:
            file_type = 'other'
        self.loaded_file_path = file_path
        self.loaded_file_type = file_type
        self.show_file_preview(file_path, file_type)

    def on_multimodal_analysis_complete(self, result):
        self.hide_progress()
        self.set_buttons_enabled(True)
        self.add_message("Hybrix", f"<b>📎 Análise Multimodal Completa:</b><br><br>{result}", False)

    def on_multimodal_analysis_error(self, error):
        self.hide_progress()
        self.set_buttons_enabled(True)
        self.add_message("Hybrix", f"❌ Erro na análise multimodal: {error}", False)

    def show_file_preview(self, file_path, file_type):
        self.file_preview_widget.setVisible(True)
        if file_type == 'image':
            pixmap = QPixmap(file_path).scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.preview_icon.setPixmap(pixmap)
            self.preview_label.setText(f"Imagem carregada: <b>{os.path.basename(file_path)}</b>")
        elif file_type == 'pdf':
            self.preview_icon.setText("📄")
            self.preview_label.setText(f"PDF carregado: <b>{os.path.basename(file_path)}</b>")
            self.preview_icon.setPixmap(QPixmap())
        elif file_type == 'video':
            self.preview_icon.setText("🎬")
            self.preview_label.setText(f"Vídeo carregado: <b>{os.path.basename(file_path)}</b>")
            self.preview_icon.setPixmap(QPixmap())
        else:
            self.preview_icon.setText("📎")
            self.preview_label.setText(f"Arquivo carregado: <b>{os.path.basename(file_path)}</b>")
            self.preview_icon.setPixmap(QPixmap())

    def clear_file_preview(self):
        self.file_preview_widget.setVisible(False)
        self.preview_icon.clear()
        self.preview_label.clear()
        self.loaded_file_path = None
        self.loaded_file_type = None

    def multimodal_analysis_with_prompt(self, image_path, user_prompt):
        try:
            import base64
            import requests
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            file_ext = os.path.splitext(image_path)[1].lower()
            is_pdf = file_ext == '.pdf'
            if is_pdf:
                ocr_text = self.ocr_pdf(image_path)
                prompt = f"{user_prompt}\n\nTexto extraído via OCR do PDF:\n{ocr_text}\n\nForneça uma análise detalhada, insights e recomendações."
            else:
                prompt = f"{user_prompt}\n\nAnalise a imagem fornecida considerando a orientação acima. Seja detalhado, profissional e forneça insights."
            if is_pdf:
                data = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }]
                }
            else:
                data = {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }]
                }
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                headers=headers,
                json=data,
                params={'key': GOOGLE_API_KEY},
                timeout=60
            )
            if response.status_code == 200:
                response_data = response.json()
                if 'candidates' in response_data and response_data['candidates']:
                    analysis = response_data['candidates'][0]['content']['parts'][0]['text']
                    file_info = f"""
<b>📁 Informações do Arquivo:</b>
• Nome: {os.path.basename(image_path)}
• Tipo: {file_ext.upper() if file_ext else 'Desconhecido'}
• Tamanho: {os.path.getsize(image_path) / 1024:.1f} KB
• Data: {datetime.fromtimestamp(os.path.getmtime(image_path)).strftime('%d/%m/%Y %H:%M')}
<b>🔍 Análise Multimodal Completa:</b>
{analysis}
"""
                    return file_info
                else:
                    raise Exception("Resposta inesperada da API Gemini")
            else:
                raise Exception(f"Erro na API Gemini: {response.status_code}")
        except Exception as e:
            try:
                ocr_result = self.universal_ocr(image_path)
                return f"""
<b>⚠️ Análise Multimodal Falhou - Fallback OCR:</b>
Erro: {str(e)}
<b>📄 Texto Extraído (OCR):</b>
{ocr_result}
"""
            except:
                return f"❌ Erro completo na análise: {str(e)}"

    def universal_ocr(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            text = self.ocr_image(file_path)
        elif ext == '.pdf':
            text = self.ocr_pdf(file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            text = self.ocr_video(file_path)
        else:
            raise Exception("Formato não suportado para OCR.")
        return text

    def ocr_image(self, image_path):
        try:
            import cv2
            import pytesseract
        except ImportError:
            raise Exception("Dependências OCR não instaladas. Instale: pip install opencv-python pytesseract")
        
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Não foi possível carregar a imagem")
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂãÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ.,;:!?()[]{}"\'-_/@#$%&*+=<>|~` '
        
        text = pytesseract.image_to_string(rgb, config=custom_config, lang='por+eng')
        return text.strip()

    def ocr_pdf(self, pdf_path):
        try:
            import fitz
            import cv2
            import pytesseract
            import numpy as np
            from pdf2image import convert_from_path
        except ImportError:
            raise Exception("Dependências PDF não instaladas. Instale: pip install PyMuPDF opencv-python pytesseract pdf2image")
        
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                t = page.get_text()
                if t.strip():
                    text += t + '\n'
            doc.close()
        except:
            pass
        
        if not text.strip():
            try:
                images = convert_from_path(pdf_path)
                for i, img in enumerate(images):
                    img_np = np.array(img)
                    rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGB)
                    
                    custom_config = r'--oem 3 --psm 6'
                    page_text = pytesseract.image_to_string(rgb, config=custom_config, lang='por+eng')
                    text += f"[Página {i+1}]\n{page_text}\n\n"
            except Exception as e:
                text += f"Erro processando PDF: {str(e)}\n"
        
        return text.strip()

    def ocr_video(self, video_path):
        try:
            import cv2
            import pytesseract
        except ImportError:
            raise Exception("Dependências OCR não instaladas. Instale: pip install opencv-python pytesseract")
        
        cap = cv2.VideoCapture(video_path)
        text = ""
        frame_count = 0
        
        if not cap.isOpened():
            raise Exception("Não foi possível abrir o vídeo")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 30:
                break
            
            if frame_count % 10 == 0:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    custom_config = r'--oem 3 --psm 6'
                    frame_text = pytesseract.image_to_string(rgb, config=custom_config, lang='por+eng')
                    
                    if frame_text.strip():
                        text += f"[Frame {frame_count}]\n{frame_text.strip()}\n\n"
                except Exception as e:
                    text += f"[Frame {frame_count}] Erro: {str(e)}\n"
            
            frame_count += 1
        
        cap.release()
        return text.strip()

    def is_simple_table(self, text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) < 2:
            return False
        for line in lines:
            if ',' not in line:
                return False
        return True

    def clean_ai_graph_response(self, text):
        text = re.sub(r'```python[\s\S]*?(plt\.show\(\)|plt\.bar|plt\.pie|plt\.plot|import matplotlib|import seaborn|import plotly)[\s\S]*?```', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(use o seguinte código[\s\S]*?para gerar o gráfico[\s\S]*?\.|você pode usar matplotlib[\s\S]*?\.|rode este código[\s\S]*?\.|utilize o seguinte código[\s\S]*?\.|para gerar o gráfico, use[\s\S]*?\.|você pode gerar o gráfico usando[\s\S]*?\.|utilize matplotlib[\s\S]*?\.|utilize plotly[\s\S]*?\.|segue o código[\s\S]*?\.|segue exemplo de código[\s\S]*?\.|copie e cole o código[\s\S]*?\.|execute o código[\s\S]*?\.|gere o gráfico manualmente[\s\S]*?\.|crie o gráfico manualmente[\s\S]*?\.|faça o gráfico no excel[\s\S]*?\.|faça o gráfico no google sheets[\s\S]*?\.|faça o gráfico no canva[\s\S]*?\.|faça o gráfico em ferramenta online[\s\S]*?\.|utilize ferramentas como[\s\S]*?\.|você pode usar ferramentas como[\s\S]*?\.|você pode usar o excel[\s\S]*?\.|você pode usar o google sheets[\s\S]*?\.|você pode usar o canva[\s\S]*?\.|você pode usar chart maker[\s\S]*?\.|utilize chart maker[\s\S]*?\.|utilize o excel[\s\S]*?\.|utilize o google sheets[\s\S]*?\.|utilize o canva[\s\S]*?\.|utilize o chart maker[\s\S]*?\.)', 'Gráfico gerado automaticamente abaixo:', text, flags=re.IGNORECASE)
        text = re.sub(r'(segue o código[\s\S]*?\.|segue exemplo de código[\s\S]*?\.)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```[\s\S]*?(plt\.show\(\)|plt\.bar|plt\.pie|plt\.plot|import matplotlib|import seaborn|import plotly)[\s\S]*?```', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(para criar o gráfico[\s\S]*?\.|você pode usar[\s\S]*?\.|crie o gráfico[\s\S]*?\.|gere o gráfico[\s\S]*?\.|utilize ferramentas[\s\S]*?\.|ferramentas online[\s\S]*?\.|faça o gráfico[\s\S]*?\.)', 'Gráfico gerado automaticamente abaixo:', text, flags=re.IGNORECASE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

class HybrixSplashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        
        self.setPixmap(QPixmap(600, 400))
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | 
                          Qt.WindowType.WindowStaysOnTopHint)
        self.setEnabled(False)
        
        self.angle = 0
        self.message = "Iniciando Hybrix AI..."
        self.is_animating = False

        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(100)
        
        self.show()
    
    def update_animation(self):
        if self.is_animating:
            return
            
        self.is_animating = True
        self.angle = (self.angle + 15) % 360
        self.update()
        self.is_animating = False
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        try:
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor(26, 26, 26))
            gradient.setColorAt(1, QColor(40, 40, 40))
            painter.fillRect(self.rect(), gradient)
            
            self.draw_spinner(painter)
            self.draw_icon(painter)
            self.draw_text(painter)
        finally:
            painter.end()
    
    def draw_spinner(self, painter):
        center = self.rect().center()
        spinner_rect = QRect(0, 0, 120, 120)
        spinner_rect.moveCenter(center - QPoint(0, 50))
        
        pen = QPen(QColor(0, 132, 255), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(spinner_rect, self.angle * 16, 120 * 16)
    
    def draw_icon(self, painter):
        center = self.rect().center()
        icon_rect = QRect(0, 0, 80, 80)
        icon_rect.moveCenter(center - QPoint(0, 50))
        
        if os.path.exists("barco.png"):
            try:
                icon = QPixmap("barco.png").scaled(80, 80,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                painter.drawPixmap(icon_rect, icon)
                return
            except Exception:
                pass
        
        painter.setFont(QFont("Arial", 48))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(icon_rect, Qt.AlignmentFlag.AlignCenter, "⛵")
    
    def draw_text(self, painter):
        center = self.rect().center()
        
        painter.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            QRect(0, center.y() + 40, self.width(), 40),
            Qt.AlignmentFlag.AlignCenter,
            "Hybrix AI"
        )
        
        painter.setFont(QFont("Segoe UI", 12))
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(
            QRect(0, center.y() + 80, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            self.message
        )
    
def show_message(self, message):
    self.message = message
    self.update()

def close(self):
    """Fecha o splash screen suavemente"""
    self.hide()
    super().close()


class ModernMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setWindowIcon(QIcon("barco.png"))
    
    def setup_ui(self):
        self.setWindowTitle("Hybrix AI")
        self.setGeometry(100, 100, 974, 541)
        self.setStyleSheet("")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.sidebar = Sidebar()
        self.sidebar.new_chat_requested.connect(self.new_chat)
        self.sidebar.chat_selected.connect(self.load_chat)
        self.sidebar.settings_clicked.connect(self.open_settings)
        main_layout.addWidget(self.sidebar)
        self.chat_area = ChatArea()
        main_layout.addWidget(self.chat_area, 1)

    def new_chat(self):
        if self.chat_area.current_session:
            title = self.get_chat_title()
            self.save_current_session(title)
            self.sidebar.add_chat_session(title)
        self.chat_area.clear_chat()
    
    def load_chat(self, title):
        self.chat_area.clear_chat()
        try:
            if os.path.exists("hybrix_chat_history.json"):
                with open("hybrix_chat_history.json", "r", encoding="utf-8") as f:
                    history = json.load(f)
                messages = history.get(title, [])
                for msg in messages:
                    self.chat_area.add_message(msg.get('sender',''), msg.get('text',''), msg.get('is_user', False))
                self.chat_area.current_session = messages.copy()
        except Exception as e:
            print(f"Hybrix: Erro carregando histórico: {e}")
    
    def get_chat_title(self):
        for msg in self.chat_area.current_session:
            if msg['is_user']:
                title = msg['text'][:40]
                if len(msg['text']) > 40:
                    title += "..."
                return title
        return "Nova conversa com Hybrix"
    
    def closeEvent(self, event):
        if self.chat_area.current_session:
            title = self.get_chat_title()
            self.save_current_session(title)
            self.sidebar.add_chat_session(title)
        save_app_settings()
        event.accept()
    
    def save_current_session(self, title):
        try:
            history = {}
            if os.path.exists("hybrix_chat_history.json"):
                with open("hybrix_chat_history.json", "r", encoding="utf-8") as f:
                    history = json.load(f)
            history[title] = self.chat_area.current_session
            with open("hybrix_chat_history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Hybrix: Erro salvando histórico: {e}")
    
    def open_settings(self):
        QMessageBox.information(self, "Configurações", "Funcionalidade de configurações em desenvolvimento.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    
    # Criar e mostrar o splash screen
    splash = HybrixSplashScreen()
    splash.show()
    
    # Processar eventos para garantir que o splash seja exibido
    app.processEvents()
    
    # Inicializar configurações (remover se não tiver thread ainda)
    # settings_thread.start()
    
    # Criar a janela principal
    main_window = ModernMainWindow()
    
    # Fechar splash e abrir a janela principal após 2.5 segundos
    QTimer.singleShot(2500, splash.close)
    QTimer.singleShot(2500, main_window.show)
    
    sys.exit(app.exec())
