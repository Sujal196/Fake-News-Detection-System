// DOM Elements
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

// Theme Toggle
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

// Navbar Scroll
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
});

// Character Counter & Validation
textArea.addEventListener('input', function() {
    const len = this.value.length;
    charCount.textContent = len;
    
    charDot.className = 'char-dot';
    if (len >= 50) {
        charDot.classList.add('valid');
    } else if (len >= 10) {
        charDot.classList.add('warn');
    }
    
    // Auto-resize
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Example Tags
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

// Form Submission
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

// Loading State
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

// Render Results
function renderResult(data) {
    const { prediction, confidence, probabilities, explanation } = data;
    const isFake = prediction === 'Fake News';
    const confPct = Math.round(confidence * 100);
    const fPct = (probabilities['Fake News'] * 100).toFixed(1);
    const rPct = (probabilities['Real News'] * 100).toFixed(1);
    
    // Show section
    resultSection.style.display = 'block';
    
    // Smooth scroll to result
    setTimeout(() => {
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
    // Reset classes
    rGlow.className = 'result-glow ' + (isFake ? 'fake-glow' : 'real-glow');
    verdictIcon.className = 'verdict-icon ' + (isFake ? 'fake-icon' : 'real-icon');
    verdictIcon.innerHTML = isFake ? '⚠️' : '✅';
    
    verdictLabel.textContent = isFake ? 'FAKE NEWS' : 'REAL NEWS';
    verdictLabel.className = 'verdict-label ' + (isFake ? 'fake-text' : 'real-text');
    
    // Animate Ring
    ringPct.textContent = confPct + '%';
    ringProg.className = 'ring-prog ' + (isFake ? 'fake-stroke' : 'real-stroke');
    
    // Circumference of r=52 is 2 * PI * 52 ≈ 326.7
    const circ = 326.7;
    const offset = circ - (confPct / 100) * circ;
    
    // Need a tiny timeout to allow CSS transition to trigger from default
    setTimeout(() => {
        ringProg.style.strokeDashoffset = offset;
    }, 50);
    
    // Prob Bars
    fakePct.textContent = fPct + '%';
    realPct.textContent = rPct + '%';
    
    setTimeout(() => {
        fakeBar.style.width = fPct + '%';
        realBar.style.width = rPct + '%';
    }, 50);
    
    // Explanation Formatting
    let html = explanation.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');
    if (!html.startsWith('<p>')) html = '<p>' + html + '</p>';
    
    explainContent.innerHTML = html;
}

// --- Analytics Dashboard Logic ---
function showTagInsight(tag, insight) {
    document.querySelectorAll('.ad-tag').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    
    const res = document.getElementById('adInsightResult');
    if (!res) return;
    
    res.style.opacity = 0;
    setTimeout(() => {
        res.innerHTML = `<div><strong>${tag}:</strong> ${insight}</div>`;
        res.style.opacity = 1;
    }, 200);
}

// Number Counter Animation for Analytics
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
    if(document.querySelector('.analytics-dash')) {
        animateAnalyticsNumbers();
    }

    // Hamburger Menu Logic
    const hamburger = document.getElementById('hamburger');
    const navLinks = document.getElementById('navLinks');
    if(hamburger && navLinks) {
        hamburger.addEventListener('click', () => {
            navLinks.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
        
        // Close menu on link click
        navLinks.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
                hamburger.classList.remove('active');
            });
        });
    }
});

// Toast Notifications
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
