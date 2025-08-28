// popup.js
/**
 * Chrome 확장 팝업 제어 스크립트
 */

document.addEventListener('DOMContentLoaded', function() {
    const statusDiv = document.getElementById('status');
    const statusText = document.getElementById('status-text');
    const toggleButton = document.getElementById('toggle-ai');
    const refreshButton = document.getElementById('refresh-board');
    const helpButton = document.getElementById('help');
    
    let isAIEnabled = false;

    // 현재 탭에서 상태 확인
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.tabs.sendMessage(tabs[0].id, {action: 'getStatus'}, function(response) {
            if (chrome.runtime.lastError) {
                updateStatus('No game detected', false);
                return;
            }
            
            if (response) {
                updateStatus(response.message, response.gameDetected);
                isAIEnabled = response.aiEnabled;
                updateToggleButton();
            }
        });
    });

    // AI 토글 버튼
    toggleButton.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {
                action: isAIEnabled ? 'disableAI' : 'enableAI'
            }, function(response) {
                if (response) {
                    isAIEnabled = response.aiEnabled;
                    updateToggleButton();
                    updateStatus(response.message, response.gameDetected);
                }
            });
        });
    });

    // 보드 새로고침 버튼
    refreshButton.addEventListener('click', function() {
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {action: 'refreshBoard'}, function(response) {
                if (response) {
                    updateStatus(response.message, response.gameDetected);
                }
            });
        });
    });

    // 도움말 버튼
    helpButton.addEventListener('click', function() {
        chrome.tabs.create({
            url: 'https://huggingface.co/kbsooo/AlphaApple'
        });
    });

    function updateStatus(message, gameDetected) {
        statusText.textContent = message;
        statusDiv.className = gameDetected ? 'status active' : 'status inactive';
    }

    function updateToggleButton() {
        if (isAIEnabled) {
            toggleButton.textContent = '🛑 Disable AI';
            toggleButton.className = 'secondary-btn';
        } else {
            toggleButton.textContent = '🤖 Enable AI Assistant';
            toggleButton.className = 'primary-btn';
        }
    }
});