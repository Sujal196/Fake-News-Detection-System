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

document.querySelectorAll('.ex-tag').forEach(tag => {
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
            body: JSON.stringify({ text })
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        renderResult(data);

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
                html += '<div style="height:8px"></div>';
                return;
            }

            const boldified = trimmed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

            if (trimmed.startsWith('•')) {
                if (!inBulletList) {
                    html += '<ul style="margin:6px 0 6px 0;padding-left:1.2em;list-style:none;">';
                    inBulletList = true;
                }
                const content = boldified.replace(/^•\s*/, '');
                html += `<li style="margin:4px 0;padding-left:0.2em;">• ${content}</li>`;
            } else {
                if (inBulletList) {
                    html += '</ul>';
                    inBulletList = false;
                }
                html += `<p style="margin:6px 0;line-height:1.6;">${boldified}</p>`;
            }
        });

        if (inBulletList) html += '</ul>';
        return html;
    }

    explainContent.innerHTML = formatExplanation(explanation);
}

function showTagInsight(tag, insight) {
    document.querySelectorAll('.ad-tag').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');

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
