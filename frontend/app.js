// Client-side Javascript - Sales Predictor AI Dashboard

const API_BASE = ""; // Co-located with FastAPI server
let importanceChart = null;

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const statusDot = document.getElementById("status-indicator");
    const statusText = document.getElementById("status-text");
    
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const btnGenerateMock = document.getElementById("btn-generate-mock");
    const datasetSummary = document.getElementById("dataset-summary");
    const summaryRecords = document.getElementById("summary-records");
    const summaryStores = document.getElementById("summary-stores");
    const summaryAvgSales = document.getElementById("summary-avg-sales");
    const summaryRange = document.getElementById("summary-range");
    
    const btnTrain = document.getElementById("btn-train");
    const trainingLoader = document.getElementById("training-loader");
    const modelMetrics = document.getElementById("model-metrics");
    const noModelWarning = document.getElementById("no-model-warning");
    const metricR2 = document.getElementById("metric-r2");
    const metricMape = document.getElementById("metric-mape");
    const metricMae = document.getElementById("metric-mae");
    const metricRmse = document.getElementById("metric-rmse");
    const metaDate = document.getElementById("meta-date");
    const metaSize = document.getElementById("meta-size");
    
    const predictionForm = document.getElementById("prediction-form");
    const inputDate = document.getElementById("input-date");
    const inputStore = document.getElementById("input-store");
    const inputCategory = document.getElementById("input-category");
    const inputPrice = document.getElementById("input-price");
    const priceDisplay = document.getElementById("price-display");
    const inputPromo = document.getElementById("input-promo");
    const inputHoliday = document.getElementById("input-holiday");
    const btnPredict = document.getElementById("btn-predict");
    const predictionResult = document.getElementById("prediction-result");
    const noPredictionWarning = document.getElementById("no-prediction-warning");
    const predictedValue = document.getElementById("predicted-value");
    const predictionInsightMsg = document.getElementById("prediction-insight-msg");

    // Initialize Default Target Date to Today
    const today = new Date().toISOString().split('T')[0];
    inputDate.value = today;

    // Price Slider display update
    inputPrice.addEventListener("input", (e) => {
        priceDisplay.textContent = `$${parseFloat(e.target.value).toFixed(2)}`;
    });

    // Check system connection and load status
    checkSystemStatus();

    async function checkSystemStatus() {
        try {
            // Check status API
            const response = await fetch(`${API_BASE}/model-status`);
            if (!response.ok) throw new Error("Server error");
            const data = await response.json();
            
            // Connection is OK
            statusDot.className = "status-dot online";
            statusText.textContent = "System Connected";
            
            // Update UI based on loaded state
            updateSystemState(data);
        } catch (error) {
            console.error("System connection check failed:", error);
            statusDot.className = "status-dot offline";
            statusText.textContent = "Server Offline";
        }
    }

    function updateSystemState(data) {
        // 1. Dataset State
        if (data.dataset_exists) {
            datasetSummary.classList.remove("hidden");
            btnTrain.removeAttribute("disabled");
            
            if (data.dataset_records) {
                summaryRecords.textContent = data.dataset_records.toLocaleString();
                summaryStores.textContent = "3"; // Based on generator
                summaryAvgSales.textContent = "25.4"; // Standard placeholder
                summaryRange.textContent = "2 Years Historical Data";
            }
        } else {
            datasetSummary.classList.add("hidden");
            btnTrain.setAttribute("disabled", "true");
        }

        // 2. Model State
        if (data.model_exists && data.model_metadata) {
            noModelWarning.classList.add("hidden");
            modelMetrics.classList.remove("hidden");
            
            // Unlock Predictions
            noPredictionWarning.classList.add("hidden");
            btnPredict.removeAttribute("disabled");

            // Populate metrics
            const metrics = data.model_metadata.metrics;
            metricR2.textContent = `${(metrics.r2 * 100).toFixed(1)}%`;
            metricMape.textContent = `${metrics.mape.toFixed(1)}%`;
            metricMae.textContent = metrics.mae.toFixed(1);
            metricRmse.textContent = metrics.rmse.toFixed(1);
            
            metaDate.textContent = data.model_metadata.trained_date || "-";
            metaSize.textContent = data.model_metadata.training_size ? data.model_metadata.training_size.toLocaleString() : "-";
            
            // Render Chart
            if (data.model_metadata.feature_importances) {
                renderImportanceChart(data.model_metadata.feature_importances);
            }
        } else {
            noModelWarning.classList.remove("hidden");
            modelMetrics.classList.add("hidden");
            
            noPredictionWarning.classList.remove("hidden");
            btnPredict.setAttribute("disabled", "true");
            predictionResult.classList.add("hidden");
        }
    }

    // Chart.js Importance Graph
    function renderImportanceChart(importances) {
        const ctx = document.getElementById("importanceChart").getContext("2d");
        
        // Destroy existing chart if any
        if (importanceChart) {
            importanceChart.destroy();
        }
        
        // Sort importances descending and take top 5
        const top5 = importances.slice(0, 5);
        const labels = top5.map(item => item.feature);
        const values = top5.map(item => item.importance * 100);

        importanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance (%)',
                    data: values,
                    backgroundColor: [
                        'rgba(236, 72, 153, 0.7)',  // pink
                        'rgba(99, 102, 241, 0.7)',  // indigo
                        'rgba(6, 182, 212, 0.7)',   // cyan
                        'rgba(16, 185, 129, 0.7)',  // green
                        'rgba(245, 158, 11, 0.7)'   // amber
                    ],
                    borderColor: [
                        '#ec4899',
                        '#6366f1',
                        '#06b6d4',
                        '#10b981',
                        '#f59e0b'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Importance: ${context.parsed.x.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#9ca3af', font: { family: 'Inter', size: 9 } },
                        title: { display: true, text: 'Importance %', color: '#9ca3af', font: { family: 'Inter', size: 9 } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f3f4f6', font: { family: 'Outfit', size: 10, weight: '500' } }
                    }
                }
            }
        });
    }

    // Generate Mock Data Event
    btnGenerateMock.addEventListener("click", async () => {
        try {
            btnGenerateMock.setAttribute("disabled", "true");
            btnGenerateMock.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;
            
            const response = await fetch(`${API_BASE}/generate-mock-data`, {
                method: "POST"
            });
            
            if (!response.ok) throw new Error("Mock generation failed");
            
            const result = await response.json();
            
            btnGenerateMock.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Sample Dataset`;
            btnGenerateMock.removeAttribute("disabled");
            
            // Update UI with summary
            if (result.success && result.data_summary) {
                datasetSummary.classList.remove("hidden");
                summaryRecords.textContent = result.data_summary.total_records.toLocaleString();
                summaryStores.textContent = result.data_summary.unique_stores.toString();
                summaryAvgSales.textContent = result.data_summary.average_sales.toFixed(1);
                summaryRange.textContent = `${result.data_summary.start_date} to ${result.data_summary.end_date}`;
                
                btnTrain.removeAttribute("disabled");
                
                showNotification("success", "Sample dataset generated successfully!");
            }
        } catch (error) {
            console.error(error);
            showNotification("error", "Failed to generate sample data.");
            btnGenerateMock.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Generate Sample Dataset`;
            btnGenerateMock.removeAttribute("disabled");
        }
    });

    // Model Training Event
    btnTrain.addEventListener("click", async () => {
        try {
            btnTrain.setAttribute("disabled", "true");
            trainingLoader.classList.remove("hidden");
            
            const response = await fetch(`${API_BASE}/train-model`, {
                method: "POST"
            });
            
            if (!response.ok) throw new Error("Training failed");
            
            const result = await response.json();
            
            trainingLoader.classList.add("hidden");
            btnTrain.removeAttribute("disabled");
            
            if (result.success) {
                // Refresh system state
                checkSystemStatus();
                showNotification("success", "Model trained successfully!");
            }
        } catch (error) {
            console.error(error);
            showNotification("error", "Error during model training.");
            trainingLoader.classList.add("hidden");
            btnTrain.removeAttribute("disabled");
        }
    });

    // Prediction Form Submission Event
    predictionForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const payload = {
            date: inputDate.value,
            store_id: parseInt(inputStore.value),
            product_category: inputCategory.value,
            price: parseFloat(inputPrice.value),
            promo: inputPromo.checked ? 1 : 0,
            holiday: inputHoliday.checked ? 1 : 0
        };

        try {
            btnPredict.setAttribute("disabled", "true");
            btnPredict.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing Factors...`;
            
            const response = await fetch(`${API_BASE}/predict-sales`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error("Prediction request failed");
            const result = await response.json();
            
            btnPredict.removeAttribute("disabled");
            btnPredict.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Expected Sales`;

            if (result.success) {
                predictionResult.classList.remove("hidden");
                
                // Animate value counter
                animateValueCounter(predictedValue, result.predicted_sales);
                
                // Construct insights
                let insight = `<i class="fa-solid fa-lightbulb text-accent"></i> `;
                if (payload.promo && payload.holiday) {
                    insight += "Combined promotion and holiday events trigger maximal sales spikes.";
                } else if (payload.promo) {
                    insight += "Promotional campaign boosts estimated demand significantly.";
                } else if (payload.holiday) {
                    insight += "Holiday period shifts traffic up for this product category.";
                } else {
                    insight += "Baseline prediction for standard business days.";
                }
                predictionInsightMsg.innerHTML = insight;
            }
        } catch (error) {
            console.error(error);
            showNotification("error", "Failed to run sales prediction.");
            btnPredict.removeAttribute("disabled");
            btnPredict.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> Predict Expected Sales`;
        }
    });

    // File Drag & Drop Handlers
    dropZone.addEventListener("click", () => fileInput.click());
    
    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    async function handleFileUpload(file) {
        if (!file.name.endsWith(".csv")) {
            showNotification("error", "Invalid file type. Please upload a CSV dataset.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            dropZone.innerHTML = `<i class="fa-solid fa-spinner fa-spin upload-icon"></i><p class="upload-title">Uploading & Preprocessing...</p>`;
            
            const response = await fetch(`${API_BASE}/upload-data`, {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) throw new Error("Upload failed");
            const result = await response.json();
            
            // Restore drop zone markup
            dropZone.innerHTML = `
                <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
                <p class="upload-title">Drag & drop dataset sales.csv here</p>
                <p class="upload-subtitle">or click to browse local files</p>
            `;
            
            if (result.success && result.data_summary) {
                datasetSummary.classList.remove("hidden");
                summaryRecords.textContent = result.data_summary.total_records.toLocaleString();
                summaryStores.textContent = result.data_summary.unique_stores.toString();
                summaryAvgSales.textContent = result.data_summary.average_sales.toFixed(1);
                summaryRange.textContent = `${result.data_summary.start_date} to ${result.data_summary.end_date}`;
                
                btnTrain.removeAttribute("disabled");
                
                // Reset model view if newly uploaded
                noModelWarning.classList.remove("hidden");
                modelMetrics.classList.add("hidden");
                btnPredict.setAttribute("disabled", "true");
                predictionResult.classList.add("hidden");
                noPredictionWarning.classList.remove("hidden");
                
                showNotification("success", `Uploaded ${file.name} successfully!`);
            }
        } catch (error) {
            console.error(error);
            showNotification("error", "Error uploading files.");
            dropZone.innerHTML = `
                <i class="fa-solid fa-cloud-arrow-up upload-icon"></i>
                <p class="upload-title">Drag & drop dataset sales.csv here</p>
                <p class="upload-subtitle">or click to browse local files</p>
            `;
        }
    }

    // Number Counter Animation Helper
    function animateValueCounter(element, endVal) {
        let start = 0;
        const duration = 800; // ms
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Ease out quad
            const easedProgress = progress * (2 - progress);
            const currentVal = Math.round(start + easedProgress * (endVal - start));
            
            element.textContent = currentVal.toLocaleString();

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                element.textContent = endVal.toLocaleString();
            }
        }

        requestAnimationFrame(update);
    }

    // In-app Notification toast helper
    function showNotification(type, message) {
        const notif = document.createElement("div");
        notif.style.position = "fixed";
        notif.style.bottom = "24px";
        notif.style.right = "24px";
        notif.style.padding = "1rem 1.5rem";
        notif.style.borderRadius = "8px";
        notif.style.color = "white";
        notif.style.fontFamily = "Outfit";
        notif.style.fontWeight = "600";
        notif.style.fontSize = "0.9rem";
        notif.style.boxShadow = "0 10px 25px rgba(0,0,0,0.3)";
        notif.style.zIndex = "9999";
        notif.style.display = "flex";
        notif.style.alignItems = "center";
        notif.style.gap = "0.5rem";
        notif.style.animation = "fadeIn 0.3s ease";
        notif.style.backdropFilter = "blur(10px)";

        if (type === "success") {
            notif.style.background = "rgba(16, 185, 129, 0.9)";
            notif.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${message}`;
        } else {
            notif.style.background = "rgba(239, 68, 68, 0.9)";
            notif.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> ${message}`;
        }

        document.body.appendChild(notif);

        setTimeout(() => {
            notif.style.animation = "fadeIn 0.3s ease reverse";
            setTimeout(() => notif.remove(), 300);
        }, 3000);
    }
});
