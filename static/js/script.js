// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const newsText = document.getElementById('newsText');
const charCount = document.getElementById('charCount');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const resultLabel = document.getElementById('resultLabel');
const confidenceBadge = document.getElementById('confidenceBadge');
const fakeBar = document.getElementById('fakeBar');
const realBar = document.getElementById('realBar');
const fakeProb = document.getElementById('fakeProb');
const realProb = document.getElementById('realProb');

// Character counter
newsText.addEventListener('input', () => {
    const count = newsText.value.length;
    charCount.textContent = count;
    
    // Change color based on length
    if (count < 50) {
        charCount.style.color = '#e53e3e';
    } else if (count < 100) {
        charCount.style.color = '#ed8936';
    } else {
        charCount.style.color = '#48bb78';
    }
});

// Form submission
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = newsText.value.trim();
    
    if (!text) {
        showNotification('Please enter some text to analyze', 'error');
        return;
    }
    
    if (text.length < 10) {
        showNotification('Please enter at least 10 characters for accurate analysis', 'error');
        return;
    }
    
    await analyzeNews(text);
});

// Analyze news function
async function analyzeNews(text) {
    // Show loading state
    setLoadingState(true);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) {
            throw new Error('Analysis failed');
        }
        
        const result = await response.json();
        displayResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Failed to analyze text. Please try again.', 'error');
    } finally {
        setLoadingState(false);
    }
}

// Display result
function displayResult(result) {
    const { prediction, confidence, probabilities, explanation } = result;
    
    // Set result label and style
    resultLabel.textContent = prediction;
    resultLabel.className = 'result-label ' + (prediction === 'Fake News' ? 'fake' : 'real');
    
    // Set confidence
    confidenceBadge.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    
    // Set probability bars and values
    const fakeProbValue = probabilities['Fake News'] * 100;
    const realProbValue = probabilities['Real News'] * 100;
    
    fakeBar.style.width = fakeProbValue + '%';
    realBar.style.width = realProbValue + '%';
    
    fakeProb.textContent = fakeProbValue.toFixed(1) + '%';
    realProb.textContent = realProbValue.toFixed(1) + '%';
    
    // Display explanation
    displayExplanation(prediction, explanation);
    
    // Show result section with animation
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Add entrance animation
    resultSection.style.animation = 'none';
    setTimeout(() => {
        resultSection.style.animation = 'slideIn 0.5s ease-out';
    }, 10);
}

// Display explanation
function displayExplanation(prediction, explanation) {
    const explanationSection = document.getElementById('explanationSection');
    const explanationContent = document.getElementById('explanationContent');
    
    // Set explanation content with proper formatting
    explanationContent.innerHTML = formatExplanation(explanation);
    
    // Add appropriate class based on prediction
    explanationContent.className = 'explanation-content ' + (prediction === 'Fake News' ? 'fake-news' : 'real-news');
    
    // Show the explanation section
    explanationSection.style.display = 'block';
}

// Format explanation text for better display
function formatExplanation(explanation) {
    // Convert newlines to paragraphs
    let formatted = explanation.replace(/\n\n/g, '</p><p>');
    
    // Handle single newlines
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Wrap in paragraphs
    if (!formatted.startsWith('<p>')) {
        formatted = '<p>' + formatted + '</p>';
    }
    
    // Convert markdown-style bold to HTML
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Convert emoji indicators to keep them
    formatted = formatted.replace(/📊|🏛️|📈|📝|✅|⚠️|🚨|🕵️|❓|😮|📰|❌/g, (match) => match);
    
    return formatted;
}

// Set loading state
function setLoadingState(isLoading) {
    if (isLoading) {
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').style.display = 'none';
        analyzeBtn.querySelector('.btn-loading').style.display = 'inline-flex';
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.querySelector('.btn-text').style.display = 'inline';
        analyzeBtn.querySelector('.btn-loading').style.display = 'none';
    }
}

// Show notification (simple implementation)
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
        max-width: 300px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    `;
    
    // Set background color based on type
    if (type === 'error') {
        notification.style.background = 'linear-gradient(135deg, #e53e3e, #c53030)';
    } else if (type === 'success') {
        notification.style.background = 'linear-gradient(135deg, #48bb78, #38a169)';
    } else {
        notification.style.background = 'linear-gradient(135deg, #4299e1, #3182ce)';
    }
    
    // Add to page
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations dynamically
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Example text buttons for testing
document.addEventListener('DOMContentLoaded', () => {
    // Add example buttons to the form
    const exampleButtons = document.createElement('div');
    exampleButtons.className = 'example-buttons';
    exampleButtons.innerHTML = `
        <p style="margin-bottom: 10px; color: #718096; font-size: 0.9rem;">Try these examples:</p>
        <div style="display: flex; gap: 10px; flex-wrap: wrap;">
            <button type="button" class="example-btn" data-example="fake">Fake Example</button>
            <button type="button" class="example-btn" data-example="real">Real Example</button>
        </div>
    `;
    
    // Add styles for example buttons
    const exampleStyle = document.createElement('style');
    exampleStyle.textContent = `
        .example-buttons {
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
        }
        
        .example-btn {
            background: #f7fafc;
            border: 2px solid #e2e8f0;
            color: #4a5568;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .example-btn:hover {
            background: #edf2f7;
            border-color: #cbd5e0;
            transform: translateY(-1px);
        }
    `;
    document.head.appendChild(exampleStyle);
    
    // Insert after the form
    predictionForm.appendChild(exampleButtons);
    
    // Add click handlers
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const exampleType = btn.dataset.example;
            let exampleText = '';
            
            if (exampleType === 'fake') {
                exampleText = "Celebrity reveals shocking secret cure that doctors don't want you to know. This miracle treatment can cure any disease in just 24 hours! Big pharma is trying to hide this from you!";
            } else {
                exampleText = "Researchers at the University of California have published a peer-reviewed study showing promising results in cancer treatment trials. The research, which analyzed data from 500 patients over two years, demonstrates a 30% improvement in response rates.";
            }
            
            newsText.value = exampleText;
            charCount.textContent = exampleText.length;
            charCount.style.color = '#48bb78';
            
            // Scroll to form
            predictionForm.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        });
    });
});

// Auto-resize textarea
newsText.addEventListener('input', () => {
    newsText.style.height = 'auto';
    newsText.style.height = newsText.scrollHeight + 'px';
});

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    newsText.dispatchEvent(new Event('input'));
});
