const form = document.getElementById('predictionForm');
const textArea = document.getElementById('newsText');
const charCount = document.getElementById('charCount');
const charDot = document.getElementById('charDot');
const btnLoading = document.getElementById('btnLoading');
const btnText = document.getElementById('btnText');
const submitBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const rGlow = document.getElementById('rGlow');
const verdictIcon = document.getElementById('verdictIcon');
const verdictLabel = document.getElementById('verdictLabel');
const ringProg = document.getElementById('ringProg');
const ringPct = document.getElementById('ringPct');
const fakeBar = document.getElementById('fakeBar');
const realBar = document.getElementById('realBar');
const fakePct = document.getElementById('fakePct');
const realPct = document.getElementById('realPct');
const explainContent = document.getElementById('explainContent');
const navbar = document.getElementById('navbar');

const root = document.documentElement;
const savedTheme = localStorage.getItem('theme');

let sessionId = sessionStorage.getItem('truthscan_session_id');
if (!sessionId) {
    sessionId = 'sess_' + Math.random().toString(36).substring(2, 15) + '_' + Date.now();
    sessionStorage.setItem('truthscan_session_id', sessionId);
}

function formatIST(date) {
    if (!date || !(date instanceof Date) || isNaN(date.getTime())) {
        return '';
    }
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const ampm = date.getHours() >= 12 ? 'PM' : 'AM';
    const displayHour = date.getHours() % 12 || 12;
    return `${String(displayHour).padStart(2, '0')}:${minutes} ${ampm} IST`;
}

if (savedTheme) {
    root.setAttribute('data-theme', savedTheme);
} else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    root.setAttribute('data-theme', 'light');
} else {
    root.setAttribute('data-theme', 'dark');
}

document.querySelectorAll('.theme-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const currentTheme = root.getAttribute('data-theme') || 'dark';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        root.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
    });
});

window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

textArea.addEventListener('input', function () {
    const len = this.value.length;
    charCount.textContent = len;

    charDot.className = 'char-dot';
    if (len >= 50) {
        charDot.classList.add('valid');
    } else if (len >= 10) {
        charDot.classList.add('warn');
    }

    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

document.querySelectorAll('.ex-tag[data-example]').forEach(tag => {
    tag.addEventListener('click', () => {
        const type = tag.dataset.example;
        if (type === 'fake') {
            textArea.value = "SHOCKING: Secret cure doctors don't want you to know! This miracle pill cures all diseases overnight. Big pharma conspiracy revealed as alien technology is found in vaccines.";
        } else {
            textArea.value = "Scientists at Harvard University published a new peer-reviewed study in the Journal of Medicine showing clinical trial results with 85 percent improvement in patient outcomes after the new treatment protocol.";
        }
        textArea.dispatchEvent(new Event('input'));
    });
});

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = textArea.value.trim();

    if (text.length < 10) {
        showToast('Please enter at least 10 characters for accurate analysis.', 'error');
        return;
    }

    setLoading(true);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, session_id: sessionId })
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();

        renderResult(data);
        textArea.value = '';
        textArea.dispatchEvent(new Event('input'));
        await new Promise(resolve => setTimeout(resolve, 1000));
        await loadHistory();
        await new Promise(resolve => setTimeout(resolve, 1000));
        await loadHistory();

    } catch (err) {
        console.error(err);
        showToast('Failed to connect to the AI engine. Please try again.', 'error');
    } finally {
        setLoading(false);
    }
});

function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
        btnText.style.display = 'none';
        btnLoading.style.display = 'flex';
        btnLoading.style.alignItems = 'center';
        btnLoading.style.gap = '8px';
    } else {
        btnText.style.display = 'flex';
        btnLoading.style.display = 'none';
    }
}

function renderResult(data) {
    const { prediction, confidence, probabilities, explanation } = data;
    const isFake = prediction === 'Fake News';
    const confPct = Math.round(confidence * 100);
    const fPct = (probabilities['Fake News'] * 100).toFixed(1);
    const rPct = (probabilities['Real News'] * 100).toFixed(1);

    resultSection.style.display = 'block';

    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    rGlow.className = 'result-glow ' + (isFake ? 'fake-glow' : 'real-glow');
    verdictIcon.className = 'verdict-icon ' + (isFake ? 'fake-icon' : 'real-icon');
    verdictIcon.innerHTML = isFake ? '⚠️' : '✅';

    verdictLabel.textContent = isFake ? 'FAKE NEWS' : 'REAL NEWS';
    verdictLabel.className = 'verdict-label ' + (isFake ? 'fake-text' : 'real-text');

    ringPct.textContent = confPct + '%';
    ringProg.className = 'ring-prog ' + (isFake ? 'fake-stroke' : 'real-stroke');

    const circ = 326.7;
    const offset = circ - (confPct / 100) * circ;

    setTimeout(() => {
        ringProg.style.strokeDashoffset = offset;
    }, 50);

    fakePct.textContent = fPct + '%';
    realPct.textContent = rPct + '%';

    setTimeout(() => {
        fakeBar.style.width = fPct + '%';
        realBar.style.width = rPct + '%';
    }, 50);

    function formatExplanation(text) {
        const lines = text.split('\n');
        let html = '';
        let inBulletList = false;

        lines.forEach(line => {
            const trimmed = line.trim();
            if (!trimmed) {
                if (inBulletList) {
                    html += '</ul>';
                    inBulletList = false;
                }
                html += '<div style="height:12px"></div>';
                return;
            }

            let parsedLine = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong style="color:var(--primary);font-weight:600;">$1</strong>');

            if (trimmed.startsWith('####')) {
                if (inBulletList) {
                    html += '</ul>';
                    inBulletList = false;
                }
                const content = parsedLine.replace(/^####\s*/, '');
                html += `<h4 style="margin:16px 0 10px 0;font-size:1rem;font-weight:600;color:var(--text);font-family:var(--font-display);text-transform:uppercase;letter-spacing:0.5px;">${content}</h4>`;
            } else if (trimmed.startsWith('###')) {
                if (inBulletList) {
                    html += '</ul>';
                    inBulletList = false;
                }
                const content = parsedLine.replace(/^###\s*/, '');
                html += `<h3 style="margin:20px 0 12px 0;font-size:1.15rem;font-weight:700;color:var(--primary);font-family:var(--font-display);border-bottom:2px solid var(--primary);padding-bottom:8px;">${content}</h3>`;
            } else if (trimmed.startsWith('•') || trimmed.startsWith('-')) {
                if (!inBulletList) {
                    html += '<ul style="margin:8px 0 8px 0;padding-left:1.5em;list-style:none;">';
                    inBulletList = true;
                }
                const content = parsedLine.replace(/^[•-]\s*/, '');
                html += `<li style="margin:8px 0;padding-left:0;line-height:1.6;position:relative;color:var(--text-muted);"><span style="color:var(--primary);margin-right:10px;font-weight:600;">▸</span>${content}</li>`;
            } else {
                if (inBulletList) {
                    html += '</ul>';
                    inBulletList = false;
                }
                html += `<p style="margin:10px 0;line-height:1.7;color:var(--text-muted);font-size:0.96rem;">${parsedLine}</p>`;
            }
        });

        if (inBulletList) html += '</ul>';
        return html;
    }

    explainContent.innerHTML = formatExplanation(explanation);
}

function showTagInsight(element, tag, insight) {
    document.querySelectorAll('.ad-tag').forEach(t => t.classList.remove('active'));
    element.classList.add('active');

    const res = document.getElementById('adInsightResult');
    if (!res) return;

    res.style.opacity = 0;
    setTimeout(() => {
        res.innerHTML = `<div><strong>${tag}:</strong> ${insight}</div>`;
        res.classList.add('active-insight');
        res.style.opacity = 1;
    }, 200);
}

function animateAnalyticsNumbers() {
    const stats = document.querySelectorAll('.ad-val.counter');
    stats.forEach(stat => {
        let text = stat.textContent;
        let suffix = '';
        if (text.includes('%')) suffix = '%';

        let target = parseFloat(text.replace(/,/g, '').replace('%', ''));
        let dec = text.includes('.') ? 1 : 0;

        const duration = 2000;
        const frames = 60;
        const step = target / frames;
        let current = 0;

        const update = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(update);
            }
            let valStr = current.toFixed(dec);
            if (dec === 0 && target > 1000) {
                valStr = parseInt(current).toLocaleString('en-US');
            }
            stat.textContent = valStr + suffix;
        }, duration / frames);
    });
}

document.addEventListener('DOMContentLoaded', () => {
    if (document.querySelector('.analytics-dash')) {
        animateAnalyticsNumbers();
    }

    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');
    if (hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            hamburger.classList.toggle('active');
        });

        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                hamburger.classList.remove('active');
            });
        });
    }

    loadHistory();

    const refreshBtn = document.getElementById('refreshHistoryBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', async () => {
            refreshBtn.style.opacity = '0.6';
            refreshBtn.style.pointerEvents = 'none';
            await loadHistory();
            refreshBtn.style.opacity = '1';
            refreshBtn.style.pointerEvents = 'auto';
        });
    }
});

function showToast(msg, type = 'success') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icon = type === 'error' ? '⚠️' : '✓';
    toast.innerHTML = `<span>${icon}</span> <span>${msg}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

async function loadHistory() {
    const historyContent = document.getElementById('historyContent');
    if (!historyContent) return;

    historyContent.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-muted);font-size:0.9rem;">Loading...</div>';

    function parseTimestamp(ts) {
        if (!ts) return new Date(0);
        let clean = ts.trim();

        if (!clean.includes('T')) {
            const parts = clean.split(' ');
            if (parts.length < 2) return new Date(0);

            const dateParts = parts[0].split('-');
            const timeParts = parts[1].split(':');

            if (dateParts.length !== 3 || timeParts.length < 2) return new Date(0);

            try {
                let d = new Date(
                    parseInt(dateParts[0]),
                    parseInt(dateParts[1]) - 1,
                    parseInt(dateParts[2]),
                    parseInt(timeParts[0]),
                    parseInt(timeParts[1]),
                    parseInt(timeParts[2] || 0)
                );
                return d;
            } catch (e) {
                return new Date(0);
            }
        } else {
            try {
                let d = new Date(clean);
                if (isNaN(d.getTime())) return new Date(0);
                return d;
            } catch (e) {
                return new Date(0);
            }
        }
    }

    function renderList(list) {
        if (!list || list.length === 0) {
            historyContent.innerHTML = `<div class="empty-history-msg"><div>No predictions stored yet.</div><div>Analyze an article above to begin.</div></div>`;
            return;
        }

        let html = '';
        list.forEach(item => {
            if (!item || !item.text) return;

            const isFake = item.prediction === 'Fake News';
            const badgeClass = isFake ? 'fake' : 'real';
            const confPct = Math.round((item.confidence || 0) * 100);

            let timeStr = '';
            try {
                const date = parseTimestamp(item.timestamp);
                timeStr = formatIST(date);
            } catch (e) {
                timeStr = '';
            }

            html += `
                <div class="history-item">
                    <div class="history-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</div>
                    <div class="history-meta">
                        <span class="history-badge ${badgeClass}">${item.prediction || 'Unknown'}</span>
                        <span class="history-conf">${confPct}%</span>
                        <span class="history-time">${timeStr}</span>
                    </div>
                </div>
            `;
        });
        historyContent.innerHTML = html;
    }

    try {
        const response = await fetch(`/history?session_id=${sessionId}&t=${Date.now()}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Failed');

        const dbData = await response.json();
        if (!dbData || dbData.length === 0) {
            renderList([]);
            return;
        }

        const cleanedData = dbData.map(item => ({
            text: item.text || '',
            prediction: item.prediction || 'Unknown',
            confidence: typeof item.confidence === 'number' ? item.confidence : 0,
            timestamp: item.timestamp || new Date().toISOString(),
            parsed_date: null
        })).filter(item => item.text.trim() !== '');

        cleanedData.forEach(item => {
            item.parsed_date = parseTimestamp(item.timestamp);
        });

        cleanedData.sort((a, b) => b.parsed_date - a.parsed_date);

        renderList(cleanedData);
    } catch (err) {
        console.error('History error:', err);
        renderList([]);
    }
}

function escapeHtml(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
