/**
 * Talking Snake - Main Application Script
 * Handles file upload, URL submission, and audio streaming
 */

// DOM Elements
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const urlInput = document.getElementById("urlInput");
const urlSubmit = document.getElementById("urlSubmit");
const textInput = document.getElementById("textInput");
const textSubmit = document.getElementById("textSubmit");
const status = document.getElementById("status");
const player = document.getElementById("player");
const audio = document.getElementById("audio");
const filename = document.getElementById("filename");
const tabs = document.querySelectorAll(".tab");
const tabContents = document.querySelectorAll(".tab-content");
const inputSection = document.getElementById("inputSection");
const processingSection = document.getElementById("processingSection");
const stopBtn = document.getElementById("stopBtn");
const pauseBtn = document.getElementById("pauseBtn");
const deviceInfo = document.getElementById("deviceInfo");
const docInfo = document.getElementById("docInfo");
const languageButtons = document.querySelectorAll("#languageButtons .style-btn");
const processingProgressBar = document.getElementById("processingProgressBar");
const streamPlayBtn = document.getElementById("streamPlayBtn");

// Custom player elements
const playerPlayBtn = document.getElementById("playerPlayBtn");
const progressBar = document.getElementById("progressBar");
const progressSlider = document.getElementById("progressSlider");
const timeDisplay = document.getElementById("timeDisplay");
const downloadBtn = document.getElementById("downloadBtn");
const deleteBtn = document.getElementById("deleteBtn");

// Constants
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

// State
let currentAbortController = null;
let selectedLanguage = "english";
let selectedStyle = "technical";
let isPaused = false;
let estimatedDuration = 0; // Estimated total duration from server
let currentDocName = ""; // Store document name for download filename
let playbackStartTime = 0; // When playback started (for tracking real elapsed time)
let playbackElapsed = 0; // Total elapsed playback time

/**
 * Format time in seconds to MM:SS
 */
function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) {
        return "0:00";
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
}

/**
 * Format a number in human-readable form (1.2K, 3.4M, etc.)
 */
function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1).replace(/\.0$/, "") + "M";
    }
    if (num >= 1000) {
        return (num / 1000).toFixed(1).replace(/\.0$/, "") + "K";
    }
    return num.toString();
}

/**
 * Get icon for document type
 */
function getDocTypeIcon(docType) {
    switch (docType) {
        case "pdf": return "fa-file-pdf";
        case "url": return "fa-link";
        case "text": return "fa-file-lines";
        default: return "fa-file";
    }
}

/**
 * Update the document info display
 */
function updateDocInfo(data) {
    const icon = getDocTypeIcon(data.doc_type);
    const docName = data.doc_name || "Document";
    const pageInfo = data.page_count ? `<span class="doc-pages"><i class="fa-solid fa-file"></i> ${data.page_count}p</span>` : "";
    const charInfo = data.total_chars ? `<span class="doc-chars"><i class="fa-solid fa-font"></i> ${formatNumber(data.total_chars)}</span>` : "";

    // Style icons mapping
    const styleIcons = {
        technical: "fa-microchip",
        narrative: "fa-book-open",
        child_narrative: "fa-child",
        news: "fa-newspaper",
        academic: "fa-graduation-cap"
    };

    // Language flags mapping
    const langFlags = {
        english: "🇬🇧",
        chinese: "🇨🇳",
        japanese: "🇯🇵",
        korean: "🇰🇷"
    };

    const styleIcon = styleIcons[selectedStyle] || "fa-microchip";
    const langFlag = langFlags[selectedLanguage] || "🇬🇧";

    docInfo.innerHTML = `
        <span class="doc-name" title="${docName}"><i class="fa-solid ${icon}"></i><span class="doc-name-text">${docName}</span></span>
        ${pageInfo}
        ${charInfo}
        <span class="doc-style" title="Style: ${selectedStyle}"><i class="fa-solid ${styleIcon}"></i></span>
        <span class="doc-lang" title="Language: ${selectedLanguage}">${langFlag}</span>
    `;
}

/**
 * Update the custom player progress bar and time display
 */
function updatePlayerProgress() {
    // For streaming WAV, browser's duration/currentTime are unreliable
    // Track real playback time ourselves
    let currentTime;
    if (playbackStartTime > 0 && !audio.paused) {
        currentTime = playbackElapsed + (Date.now() - playbackStartTime) / 1000;
    } else {
        currentTime = playbackElapsed;
    }

    // Use our estimated duration, update it if playback exceeds estimate
    let duration = estimatedDuration;
    if (currentTime > duration) {
        estimatedDuration = currentTime + 10; // Extend estimate
        duration = estimatedDuration;
    }

    // Ensure we have reasonable values
    if (duration <= 0) {
        duration = 60; // Fallback
    }

    const progress = duration > 0 ? (currentTime / duration) * 100 : 0;
    progressBar.style.width = `${Math.min(progress, 100)}%`;
    progressSlider.value = Math.min(progress, 100);
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(duration)}`;
}

/**
 * Handle seeking via the progress slider
 */
function handleSeek(e) {
    const percent = parseFloat(e.target.value);
    const duration = estimatedDuration || 60;
    const seekTime = (percent / 100) * duration;

    // Set our playback tracker
    playbackElapsed = seekTime;
    playbackStartTime = audio.paused ? 0 : Date.now();

    // Try to seek the audio (may not work well with streaming)
    try {
        audio.currentTime = seekTime;
    } catch {
        // Seeking may fail with streaming audio
    }
    updatePlayerProgress();
}

/**
 * Toggle play/pause for custom player
 */
function togglePlayerPlay() {
    if (audio.paused) {
        audio.play().catch(() => {});
    } else {
        audio.pause();
    }
}

/**
 * Update play button icon
 */
function updatePlayButton() {
    const icon = playerPlayBtn.querySelector("i");
    if (audio.paused) {
        icon.className = "fa-solid fa-play";
    } else {
        icon.className = "fa-solid fa-pause";
    }
}

/**
 * Get HTML for model state indicator
 * @param {string} state - Model state: loaded, loading, unloaded, unloading
 * @returns {string} HTML string for the model state indicator
 */
function getModelStateHtml(state) {
    const stateConfig = {
        loaded: {
            icon: "fa-circle-check",
            class: "model-loaded",
            text: "Model loaded",
            tooltip: "TTS model is loaded in memory and ready for inference"
        },
        loading: {
            icon: "fa-spinner fa-spin",
            class: "model-loading",
            text: "Loading...",
            tooltip: "TTS model is being loaded into memory"
        },
        unloaded: {
            icon: "fa-circle-xmark",
            class: "model-unloaded",
            text: "Model unloaded",
            tooltip: "TTS model is not loaded (will load on first request)"
        },
        unloading: {
            icon: "fa-spinner fa-spin",
            class: "model-unloading",
            text: "Unloading...",
            tooltip: "TTS model is being unloaded from memory"
        }
    };
    const config = stateConfig[state] || stateConfig.unloaded;
    return `<span class="model-state ${config.class}" title="${config.tooltip}"><i class="fa-solid ${config.icon}"></i> ${config.text}</span>`;
}

/**
 * Update device info display from SSE data
 * @param {Object} info - Device info object
 */
function updateDeviceInfo(info) {
    const icon = info.device === "cuda" ? "fa-microchip" : "fa-server";
    const deviceTooltip = info.device === "cuda"
        ? "GPU accelerated inference for faster audio generation"
        : "CPU-based inference (slower than GPU)";
    const gpuMemoryInfo = info.device === "cuda"
        ? `<span class="device-memory" title="GPU memory used for model and inference"><i class="fa-solid fa-memory"></i> GPU: ${info.memory_used_gb}/${info.memory_total_gb}GB</span>`
        : "";
    const ramInfo = `<span class="device-memory" title="System RAM usage"><i class="fa-solid fa-memory"></i> RAM: ${info.ram_used_gb}/${info.ram_total_gb}GB</span>`;
    // Show timing stats if available
    const timingInfo = info.seconds_per_char !== undefined
        ? `<span class="device-timing" title="Average time to generate audio per character of text"><i class="fa-solid fa-stopwatch"></i> ${info.seconds_per_char.toFixed(4)}s/char</span>`
        : "";
    // Show model state
    const modelStateInfo = getModelStateHtml(info.model_state);
    deviceInfo.innerHTML = `
        <i class="fa-solid ${icon}" title="${deviceTooltip}"></i>
        <span title="${deviceTooltip}">${info.device_name}</span>
        ${modelStateInfo}
        ${gpuMemoryInfo}
        ${ramInfo}
        ${timingInfo}
        <span class="device-ephemeral" title="Your documents are processed in memory only. Nothing is saved to disk or stored after processing."><i class="fa-solid fa-shield-halved"></i> No files stored</span>
    `;
    deviceInfo.classList.add("visible");
}

/**
 * Initialize device info SSE stream
 */
function initDeviceInfoStream() {
    const eventSource = new EventSource("/api/device-info-stream");

    eventSource.onmessage = (event) => {
        try {
            const info = JSON.parse(event.data);
            updateDeviceInfo(info);
        } catch {
            // Silently fail - device info is optional
        }
    };

    eventSource.onerror = () => {
        // On error, close and try to reconnect after a delay
        eventSource.close();
        setTimeout(initDeviceInfoStream, 5000);
    };
}

// Start device info SSE stream
initDeviceInfoStream();

// Custom player event listeners
playerPlayBtn.addEventListener("click", togglePlayerPlay);
progressSlider.addEventListener("input", handleSeek);
audio.addEventListener("play", () => {
    // Start tracking real playback time
    playbackStartTime = Date.now();
    updatePlayButton();
});
audio.addEventListener("pause", () => {
    // Save elapsed time when pausing
    if (playbackStartTime > 0) {
        playbackElapsed += (Date.now() - playbackStartTime) / 1000;
        playbackStartTime = 0;
    }
    updatePlayButton();
});
audio.addEventListener("timeupdate", updatePlayerProgress);
audio.addEventListener("ended", () => {
    // Update elapsed to match duration on completion
    if (playbackStartTime > 0) {
        playbackElapsed += (Date.now() - playbackStartTime) / 1000;
        playbackStartTime = 0;
    }
    // Ensure we show completion
    if (estimatedDuration > 0 && playbackElapsed < estimatedDuration) {
        playbackElapsed = estimatedDuration;
    }
    updatePlayButton();
    progressBar.style.width = "100%";
    timeDisplay.textContent = `${formatTime(estimatedDuration)} / ${formatTime(estimatedDuration)}`;
});
// Update duration when metadata is available
audio.addEventListener("loadedmetadata", () => {
    // If browser has a valid duration, use it instead of estimate
    if (isFinite(audio.duration) && audio.duration > 0 && audio.duration < 36000) {
        estimatedDuration = audio.duration;
    }
    updatePlayerProgress();
});
// Also check duration changes (for streaming audio)
audio.addEventListener("durationchange", () => {
    if (isFinite(audio.duration) && audio.duration > 0 && audio.duration < 36000) {
        estimatedDuration = audio.duration;
    }
    updatePlayerProgress();
});
// Log audio errors for debugging
audio.addEventListener("error", () => {
    console.error("Audio error:", audio.error?.message || "Unknown error");
});
// Show pause button when audio actually starts playing
audio.addEventListener("playing", () => {
    streamPlayBtn.classList.add("hidden");
    pauseBtn.classList.remove("hidden");
});

// Show stream play button when audio has enough data to start playing
audio.addEventListener("canplay", () => {
    // Only show if processing is still in progress (player not visible yet)
    // and audio is paused (not already playing) and pause button isn't showing
    if (!player.classList.contains("visible") && audio.paused && pauseBtn.classList.contains("hidden")) {
        streamPlayBtn.classList.remove("hidden");
    }
});

/**
 * Start streaming audio playback and enable download from cache
 * @param {string} jobId - The job ID for the audio
 */
async function startAudioStream(jobId) {
    const audioUrl = `/api/audio/${jobId}`;

    // Reset playback tracking for new stream
    playbackStartTime = 0;
    playbackElapsed = 0;

    // Set up audio source for streaming (user can click play)
    audio.src = audioUrl;
    audio.load();

    // Store job ID for download - will fetch from cache
    audio.dataset.jobId = jobId;

    // Play button will be shown by the canplay event handler
}

/**
 * Download the current audio as a WAV file
 */
function downloadAudio() {
    const jobId = audio.dataset.jobId;
    if (!jobId) {
        return;
    }

    // Create filename from document name
    let filename = currentDocName || "audio";
    filename = filename.replace(/\.[^.]+$/, "") + ".wav";

    // Use download endpoint which returns proper WAV file
    const a = document.createElement("a");
    a.href = `/api/download/${jobId}?filename=${encodeURIComponent(filename)}`;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

/**
 * Delete the current audio and reset the player
 */
function deleteAudio() {
    // Stop audio immediately
    audio.pause();

    // Add deleting animation
    player.classList.add("deleting");

    // Wait for animation to complete
    setTimeout(() => {
        // Reset audio
        audio.src = "";
        audio.currentTime = 0;

        // Clear state
        currentDocName = "";
        estimatedDuration = 0;

        // Hide player and buttons
        player.classList.remove("visible", "deleting");
        downloadBtn.classList.add("hidden");
        deleteBtn.classList.add("hidden");

        // Reset progress
        progressBar.style.width = "0%";
        progressSlider.value = 0;
        timeDisplay.textContent = "0:00 / 0:00";
        updatePlayButton();

        // Show input section again
        inputSection.classList.remove("hidden");
    }, 300);
}

/**
 * Get the currently selected language
 * @returns {string} The selected language name
 */
function getSelectedLanguage() {
    return selectedLanguage;
}

/**
 * Detect language from text based on character scripts.
 * @param {string} text - The text to analyze
 * @returns {string|null} Detected language or null if mostly ASCII/Latin
 */
function detectLanguage(text) {
    if (!text || text.length < 5) {
        return null;
    }

    let chinese = 0;
    let japanese = 0; // Hiragana + Katakana
    let korean = 0;
    let latin = 0;

    for (const char of text) {
        const code = char.charCodeAt(0);
        // CJK Unified Ideographs (shared by Chinese/Japanese)
        if (code >= 0x4e00 && code <= 0x9fff) {
            chinese++;
        }
        // Hiragana
        else if (code >= 0x3040 && code <= 0x309f) {
            japanese++;
        }
        // Katakana
        else if (code >= 0x30a0 && code <= 0x30ff) {
            japanese++;
        }
        // Hangul Syllables
        else if (code >= 0xac00 && code <= 0xd7af) {
            korean++;
        }
        // Hangul Jamo
        else if (code >= 0x1100 && code <= 0x11ff) {
            korean++;
        }
        // Basic Latin letters
        else if (
            (code >= 0x41 && code <= 0x5a) ||
            (code >= 0x61 && code <= 0x7a)
        ) {
            latin++;
        }
    }

    const total = chinese + japanese + korean + latin;
    if (total === 0) {
        return null;
    }

    // Japanese uses kanji (chinese chars) + kana, so check for kana first
    if (japanese > 0 && (japanese + chinese) / total > 0.3) {
        return "japanese";
    }
    // Korean
    if (korean / total > 0.3) {
        return "korean";
    }
    // Chinese (CJK without kana)
    if (chinese / total > 0.3) {
        return "chinese";
    }
    // Default to English for Latin text
    if (latin / total > 0.5) {
        return "english";
    }
    return null;
}

/**
 * Set the selected language, optionally marking it as auto-detected.
 * @param {string} lang - Language to select
 * @param {boolean} isAuto - Whether this was auto-detected
 */
function setLanguage(lang, isAuto = false) {
    const btn = document.querySelector(
        `#languageButtons .style-btn[data-language="${lang}"]`
    );
    if (!btn || selectedLanguage === lang) {
        return;
    }

    // Update selection state
    languageButtons.forEach((b) => {
        b.classList.remove("active", "auto-detected");
    });
    btn.classList.add("active");
    selectedLanguage = lang;

    // Visual feedback for auto-detection
    if (isAuto) {
        btn.classList.add("auto-detected");
        // Remove animation class after it completes
        setTimeout(() => btn.classList.remove("auto-detected"), 1500);
    }
}

/**
 * Get the currently selected style
 * @returns {string} The selected style ID
 */
function getSelectedStyle() {
    return selectedStyle;
}

/**
 * Show the input section and hide processing section
 */
function showInputSection() {
    inputSection.classList.remove("hidden");
    processingSection.classList.remove("visible");
}

/**
 * Show the processing section and hide input section
 */
function showProcessingSection() {
    inputSection.classList.add("hidden");
    processingSection.classList.add("visible");
    // Reset progress bar and hide buttons
    processingProgressBar.style.width = "0%";
    pauseBtn.classList.add("hidden");
    streamPlayBtn.classList.add("hidden");
}

/**
 * Show a status message to the user
 * @param {string} message - HTML message to display
 * @param {string} type - Status type: 'loading', 'error', or 'success'
 */
function showStatus(message, type) {
    status.innerHTML = message;
    status.className = `status visible ${type}`;
}

/**
 * Stop the current generation and audio playback
 */
function stopGeneration() {
    // Stop the fetch request
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    // Stop audio playback and clear source
    audio.pause();
    audio.currentTime = 0;
    audio.src = "";
    audio.load(); // Force release of audio resources

    // Reset pause state
    isPaused = false;
    updatePauseButton();

    // Hide download button, pause button, and stream play button
    downloadBtn.classList.add("hidden");
    pauseBtn.classList.add("hidden");
    streamPlayBtn.classList.add("hidden");

    // Reset progress bar
    processingProgressBar.style.width = "0%";

    showStatus('<i class="fa-solid fa-ban"></i> Generation stopped', "error");
    showInputSection();
}

// Stop audio when page is closed or navigated away
window.addEventListener("beforeunload", () => {
    audio.pause();
    audio.src = "";
});

// Also handle page hide (works better on mobile and for navigation)
window.addEventListener("pagehide", () => {
    audio.pause();
    audio.src = "";
});

/**
 * Toggle pause/play state
 */
function togglePause() {
    if (audio.paused) {
        audio.play().catch(() => {});
        isPaused = false;
    } else {
        audio.pause();
        isPaused = true;
    }
    updatePauseButton();
}

/**
 * Update pause button icon based on state
 */
function updatePauseButton() {
    const icon = pauseBtn.querySelector("i");
    if (isPaused || audio.paused) {
        icon.className = "fa-solid fa-play";
        pauseBtn.title = "Resume";
    } else {
        icon.className = "fa-solid fa-pause";
        pauseBtn.title = "Pause";
    }
}



/**
 * Get icon class for source type
 * @param {string} sourceType - The source type ("pdf", "url", "text")
 * @returns {string} Font Awesome icon class
 */
function getSourceIcon(sourceType) {
    switch (sourceType) {
        case "pdf":
            return "fa-file-pdf";
        case "url":
            return "fa-link";
        case "text":
        default:
            return "fa-keyboard";
    }
}

/**
 * Process SSE stream for progress updates
 * Sets up audio stream once job_id is received
 * @param {Response} response - Fetch response with SSE stream
 * @param {string} docName - Document name for display
 * @param {string} sourceType - Source type ("pdf", "url", "text")
 * @returns {Promise<void>}
 * @throws {Error} If stream contains an error event or fails
 */
async function processStream(response, docName, sourceType = "text") {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let lastStatus = "";
    let audioJobId = null;

    // Reset estimated duration
    estimatedDuration = 0;

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                break;
            }

            const text = decoder.decode(value, { stream: true });
            const lines = text.split("\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === "error") {
                            throw new Error(data.message || "TTS generation failed");
                        } else if (data.type === "start" && data.job_id) {
                            // Got job ID - start audio stream immediately
                            const jobId = data.job_id;
                            // Estimate audio duration from character count
                            // Typical speech is ~14 chars/sec (150 wpm, 5 chars/word)
                            if (data.total_chars) {
                                estimatedDuration = data.total_chars / 14;
                            }
                            // Display document info
                            updateDocInfo(data);
                            if (!audioJobId) {
                                audioJobId = jobId;
                                // Start streaming playback immediately
                                startAudioStream(jobId);
                            }
                            // Show generating status
                            showStatus(
                                '<span class="spinner"></span> Generating...',
                                "loading"
                            );
                            // Update progress bar
                            processingProgressBar.style.width = "5%";
                        } else if (data.type === "progress") {
                            lastStatus = data.status;
                            // Show progress percentage
                            showStatus(
                                `<span class="spinner"></span> ${data.percent}%`,
                                "loading"
                            );
                            // Update progress bar
                            processingProgressBar.style.width = `${data.percent}%`;
                        } else if (data.type === "complete") {
                            // Generation complete - show player
                            // Use actual audio duration from server if available
                            if (data.audio_duration && data.audio_duration > 0) {
                                estimatedDuration = data.audio_duration;
                            }
                            // Build filename with style and language indicators
                            const styleIcons = {
                                technical: "fa-microchip",
                                conversational: "fa-comments",
                                storytelling: "fa-book-open",
                                child_narrative: "fa-child",
                                news: "fa-newspaper",
                                academic: "fa-graduation-cap"
                            };
                            const langFlags = {
                                english: "🇬🇧",
                                chinese: "🇨🇳",
                                japanese: "🇯🇵",
                                korean: "🇰🇷"
                            };
                            const usedStyle = getSelectedStyle();
                            const usedLang = getSelectedLanguage();
                            const styleIcon = styleIcons[usedStyle] || "fa-microchip";
                            const langFlag = langFlags[usedLang] || "🇬🇧";
                            filename.innerHTML = `<i class="fa-solid ${getSourceIcon(sourceType)}"></i> ${docName} <span class="filename-meta"><i class="fa-solid ${styleIcon}" title="Style: ${usedStyle}"></i><span title="Language: ${usedLang}">${langFlag}</span></span>`;
                            currentDocName = docName;
                            // Hide stream buttons, show full player with download
                            streamPlayBtn.classList.add("hidden");
                            downloadBtn.classList.remove("hidden");
                            deleteBtn.classList.remove("hidden");
                            player.classList.add("visible");
                            // Set progress to 100%
                            processingProgressBar.style.width = "100%";
                            showInputSection();
                            showStatus(
                                `<i class="fa-solid fa-circle-check"></i> Done in ${data.total_time}s`,
                                "success"
                            );
                            updatePlayerProgress();
                        }
                    } catch (parseError) {
                        // Check if it's our thrown error or a JSON parse error
                        if (parseError.message && !parseError.message.includes("JSON")) {
                            throw parseError;
                        }
                        // Ignore JSON parse errors for partial data
                    }
                }
            }
        }
    } catch (streamError) {
        // Re-throw with more context and preserve the original cause
        const context = lastStatus ? ` (during: ${lastStatus})` : "";
        throw new Error(`Stream error${context}: ${streamError.message}`, { cause: streamError });
    }
}

/**
 * Handle file upload and TTS conversion
 * @param {File} file - The uploaded file
 */
async function handleFile(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> Please select a PDF file', "error");
        return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> File too large. Maximum size is 50MB.', "error");
        return;
    }

    showProcessingSection();
    showStatus('<span class="spinner"></span> Extracting text...', "loading");
    player.classList.remove("visible");
    downloadBtn.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("language", getSelectedLanguage());
    formData.append("style", getSelectedStyle());

    // Create abort controller for this request
    currentAbortController = new AbortController();

    try {
        const response = await fetch("/api/read-stream", {
            method: "POST",
            body: formData,
            signal: currentAbortController.signal,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to process document");
        }

        // Process stream handles both progress SSE and starting audio playback
        await processStream(response, file.name, "pdf");
    } catch (error) {
        if (error.name === "AbortError") {
            // User cancelled - already handled in stopGeneration
            return;
        }
        showStatus(`<i class="fa-solid fa-circle-exclamation"></i> ${error.message}`, "error");
        showInputSection();
    } finally {
        currentAbortController = null;
    }
}

/**
 * Handle URL submission and TTS conversion
 * @param {string} url - The URL to process
 */
async function handleUrl(url) {
    url = url.trim();

    if (!url) {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> Please enter a URL', "error");
        return;
    }

    // Validate URL format
    try {
        new URL(url);
    } catch {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> Please enter a valid URL', "error");
        return;
    }

    showProcessingSection();
    showStatus('<span class="spinner"></span> Fetching content...', "loading");
    player.classList.remove("visible");
    downloadBtn.classList.add("hidden");
    urlSubmit.disabled = true;

    // Create abort controller for this request
    currentAbortController = new AbortController();

    try {
        const response = await fetch("/api/read-url-stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                url,
                language: getSelectedLanguage(),
                style: getSelectedStyle()
            }),
            signal: currentAbortController.signal,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to process document");
        }

        // Extract filename from URL
        const urlPath = new URL(url).pathname;
        const docName = urlPath.split("/").pop() || "document";

        // Process stream handles both progress SSE and starting audio playback
        await processStream(response, docName, "url");
    } catch (error) {
        if (error.name === "AbortError") {
            // User cancelled - already handled in stopGeneration
            return;
        }
        showStatus(`<i class="fa-solid fa-circle-exclamation"></i> ${error.message}`, "error");
        showInputSection();
    } finally {
        urlSubmit.disabled = false;
        currentAbortController = null;
    }
}

/**
 * Handle text submission and TTS conversion
 * @param {string} text - The text to process
 */
async function handleText(text) {
    text = text.trim();

    if (!text) {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> Please enter some text', "error");
        return;
    }

    if (text.length > 500000) {
        showStatus('<i class="fa-solid fa-triangle-exclamation"></i> Text too long (max 500,000 characters)', "error");
        return;
    }

    showProcessingSection();
    showStatus('<span class="spinner"></span> Processing text...', "loading");
    player.classList.remove("visible");
    downloadBtn.classList.add("hidden");
    textSubmit.disabled = true;

    // Create abort controller for this request
    currentAbortController = new AbortController();

    try {
        const response = await fetch("/api/read-text-stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                text,
                language: getSelectedLanguage(),
                style: getSelectedStyle()
            }),
            signal: currentAbortController.signal,
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to process text");
        }

        // Generate document name from first few words
        const words = text.trim().split(/\s+/).slice(0, 5).join(" ");
        const docName = words.length > 30 ? words.slice(0, 30) + "..." : words;

        // Process stream handles both progress SSE and starting audio playback
        await processStream(response, docName, "text");
    } catch (error) {
        if (error.name === "AbortError") {
            // User cancelled - already handled in stopGeneration
            return;
        }
        showStatus(`<i class="fa-solid fa-circle-exclamation"></i> ${error.message}`, "error");
        showInputSection();
    } finally {
        textSubmit.disabled = false;
        currentAbortController = null;
    }
}

// Tab switching
tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
        const isAlreadyActive = tab.classList.contains("active");
        const isUploadTab = tab.dataset.tab === "upload";

        // If clicking on already-active upload tab, open file picker
        if (isAlreadyActive && isUploadTab) {
            fileInput.click();
            return;
        }

        tabs.forEach((t) => t.classList.remove("active"));
        tabContents.forEach((tc) => tc.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`${tab.dataset.tab}-tab`).classList.add("active");
    });
});

// Drag and drop handlers
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

// Click to select file
dropZone.addEventListener("click", (e) => {
    if (e.target !== fileInput) {
        fileInput.click();
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFile(fileInput.files[0]);
    }
});

// URL submission
urlSubmit.addEventListener("click", () => {
    handleUrl(urlInput.value);
});

urlInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        handleUrl(urlInput.value);
    }
});

// Text submission
textSubmit.addEventListener("click", () => {
    handleText(textInput.value);
});

// Allow Ctrl+Enter to submit text
textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        handleText(textInput.value);
    }
});

// Auto-detect language from text input
textInput.addEventListener("input", () => {
    const detected = detectLanguage(textInput.value);
    if (detected) {
        setLanguage(detected, true);
    }
});

// Stop button
stopBtn.addEventListener("click", stopGeneration);

// Stream play button (during processing)
streamPlayBtn.addEventListener("click", () => {
    audio.play().catch(() => {});
    // Hide stream play button and show pause button
    streamPlayBtn.classList.add("hidden");
    pauseBtn.classList.remove("hidden");
});

// Pause button
pauseBtn.addEventListener("click", togglePause);

// Download button
downloadBtn.addEventListener("click", downloadAudio);

// Delete button
deleteBtn.addEventListener("click", deleteAudio);

// Update pause button when audio state changes
audio.addEventListener("play", updatePauseButton);
audio.addEventListener("pause", updatePauseButton);
audio.addEventListener("ended", () => {
    isPaused = false;
    updatePauseButton();
});

// Language selection
languageButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
        setLanguage(btn.dataset.language, false);
    });
});

// Style selection
const styleButtons = document.querySelectorAll("#styleButtons .style-btn");
styleButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
        styleButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        selectedStyle = btn.dataset.style;
    });
});
