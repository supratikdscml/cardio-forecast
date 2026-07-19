document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------
    // State Variables
    // ----------------------------------------------------
    let metadataCached = null;
    let featureImportanceChart = null;
    let calibrationChart = null;

    // ----------------------------------------------------
    // Feature Label Mappers (for clean client display)
    // ----------------------------------------------------
    const featureLabels = {
        'Age': 'Patient Age',
        'RestingBP': 'Resting Blood Pressure',
        'Cholesterol': 'Serum Cholesterol',
        'FastingBS': 'Fasting Blood Sugar',
        'MaxHR': 'Max Heart Rate Achieved',
        'Oldpeak': 'ST Depression (Oldpeak)',
        'Sex': 'Sex (Male)',
        'ExerciseAngina': 'Exercise Induced Angina',
        'ChestPainType_TA': 'Typical Angina Chest Pain',
        'ChestPainType_ATA': 'Atypical Angina Chest Pain',
        'ChestPainType_NAP': 'Non-Anginal Chest Pain',
        'ChestPainType_ASY': 'Asymptomatic Chest Pain',
        'RestingECG_Normal': 'Normal Resting ECG',
        'RestingECG_ST': 'ST-T Wave Resting ECG Abnormality',
        'RestingECG_LVH': 'Left Ventricular Hypertrophy (ECG)',
        'ST_Slope_Up': 'Upsloping ST Segment',
        'ST_Slope_Flat': 'Flat ST Segment',
        'ST_Slope_Down': 'Downsloping ST Segment'
    };

    // Descriptions explaining WHY a feature drives the risk
    const driverDescriptions = {
        'ST_Slope_Flat': {
            'increases risk': 'A flat ST segment slope during peak exercise is a highly predictive indicator of myocardial ischemia.',
            'lowers risk': 'Absence of flat ST slope signals normal myocardial blood flow.'
        },
        'ST_Slope_Up': {
            'increases risk': 'An upsloping ST segment, while often normal, is associated with mild risk factors under specific workloads.',
            'lowers risk': 'An upsloping ST segment is a strong indicator of normal cardiac stress tolerance.'
        },
        'ST_Slope_Down': {
            'increases risk': 'A downsloping ST segment indicates severe coronary artery stenosis or ischemia.',
            'lowers risk': 'Absence of downsloping ST segment indicates lower likelihood of severe stenosis.'
        },
        'ChestPainType_ASY': {
            'increases risk': 'Silent ischemia (lack of chest pain symptoms despite underlying blockage) increases risk of undetected disease.',
            'lowers risk': 'Absence of asymptomatic status reduces risk.'
        },
        'ChestPainType_ATA': {
            'increases risk': 'Atypical angina chest pain indicates a mild risk of coronary arterial spasms.',
            'lowers risk': 'Atypical angina signals lower risk compared to classic asymptomatic coronary presentations.'
        },
        'ChestPainType_NAP': {
            'increases risk': 'Non-anginal chest pain, though non-cardiac, shows minor correlation with borderline stress markers.',
            'lowers risk': 'Chest pain is classified as non-cardiac, indicating a healthy heart profile.'
        },
        'ChestPainType_TA': {
            'increases risk': 'Typical angina is a classic manifestation of cardiac oxygen supply-demand mismatch.',
            'lowers risk': 'Absence of classic typical angina symptoms lowers clinical risk.'
        },
        'Oldpeak': {
            'increases risk': 'Elevated ST depression (Oldpeak) indicates a high degree of exercise-induced myocardial ischemia.',
            'lowers risk': 'A low or zero ST depression indicates normal cardiac stress recovery.'
        },
        'ExerciseAngina': {
            'increases risk': 'Angina (chest pain) induced during exertion points directly to restricted coronary artery blood flow.',
            'lowers risk': 'Lack of exercise-induced angina indicates stable cardiac arterial flow under stress.'
        },
        'MaxHR': {
            'increases risk': 'A lower maximum heart rate achieved under stress suggests cardiac fatigue or chronotropic incompetence.',
            'lowers risk': 'A robust maximum heart rate indicates excellent cardiovascular workload capacity.'
        },
        'Age': {
            'increases risk': 'Advanced age is a leading non-modifiable risk factor for arterial hardening and vessel narrowing.',
            'lowers risk': 'Younger age is associated with flexible vasculature and higher cardiac reserve.'
        },
        'Cholesterol': {
            'increases risk': 'Elevated serum cholesterol is a major driver of atherosclerotic plaque accumulation in coronary arteries.',
            'lowers risk': 'Normal or lower cholesterol levels restrict plaque development.'
        },
        'RestingBP': {
            'increases risk': 'Elevated resting blood pressure increases systemic shear stress on coronary arterial walls.',
            'lowers risk': 'Optimal resting blood pressure preserves vascular endothelial integrity.'
        },
        'Sex': {
            'increases risk': 'Male sex is clinically associated with a higher early onset rate of coronary artery disease.',
            'lowers risk': 'Female sex is statistically associated with a lower incidence of early cardiovascular events.'
        },
        'FastingBS': {
            'increases risk': 'Fasting blood sugar > 120 mg/dl is highly correlated with insulin resistance and microvascular damage.',
            'lowers risk': 'Normal blood glucose limits glycation and vascular endothelial damage.'
        },
        'RestingECG_LVH': {
            'increases risk': 'Left ventricular hypertrophy suggests long-standing systemic hypertension and thickened cardiac muscle.',
            'lowers risk': 'Absence of ventricular hypertrophy indicates healthy blood pressure history.'
        },
        'RestingECG_ST': {
            'increases risk': 'Resting ST-T wave abnormalities suggest chronic ischemic damage or early repolarization issues.',
            'lowers risk': 'Normal resting ST segment shows no signs of acute myocardial strain.'
        },
        'RestingECG_Normal': {
            'increases risk': 'Normal ECG with other high indicators can sometimes point to silent atypical abnormalities.',
            'lowers risk': 'A normal resting ECG indicates no electrical signs of ischemia at rest.'
        }
    };

    // ----------------------------------------------------
    // DOM Elements
    // ----------------------------------------------------
    const navBtnCalculator = document.getElementById('btn-calculator');
    const navBtnAnalytics = document.getElementById('btn-analytics');
    const tabPanelCalculator = document.getElementById('tab-calculator');
    const tabPanelAnalytics = document.getElementById('tab-analytics');
    const riskForm = document.getElementById('risk-form');

    // Sliders
    const sliders = [
        { id: 'input-age', valId: 'val-age' },
        { id: 'input-bp', valId: 'val-bp' },
        { id: 'input-chol', valId: 'val-chol' },
        { id: 'input-maxhr', valId: 'val-maxhr' },
        { id: 'input-oldpeak', valId: 'val-oldpeak' }
    ];

    // Load Demo Button
    const btnDemo = document.getElementById('btn-demo');

    // Results Elements
    const resPercentage = document.getElementById('res-percentage');
    const resBand = document.getElementById('res-band');
    const gaugeFillArc = document.getElementById('gauge-fill-arc');
    const gaugeNeedle = document.getElementById('gauge-needle');
    const driversList = document.getElementById('drivers-list');
    const modelsList = document.getElementById('models-list');

    // Analytics elements
    const statFriedmanP = document.getElementById('stat-friedman-p');
    const statFriedmanOutcome = document.getElementById('stat-friedman-outcome');
    const statBestModel = document.getElementById('stat-best-model');
    const statBestModelWeight = document.getElementById('stat-best-model-weight');
    const modelComparisonBars = document.getElementById('model-comparison-bars');
    const modelWeightsList = document.getElementById('model-weights-list');
    const holmTable = document.getElementById('holm-table');
    const correlationHeatmap = document.getElementById('correlation-heatmap');

    // Demo Data
    const demoPatient = {
        'Age': 58, 'Sex': 'M', 'ChestPainType': 'ASY', 'RestingBP': 145,
        'Cholesterol': 260, 'FastingBS': 1, 'RestingECG': 'ST', 'MaxHR': 122,
        'ExerciseAngina': 'Y', 'Oldpeak': 2.3, 'ST_Slope': 'Flat'
    };

    // ----------------------------------------------------
    // Tab Navigation Logic
    // ----------------------------------------------------
    function switchTab(activeTab) {
        if (activeTab === 'calculator') {
            navBtnCalculator.classList.add('active');
            navBtnAnalytics.classList.remove('active');
            tabPanelCalculator.classList.add('active');
            tabPanelAnalytics.classList.remove('active');
        } else {
            navBtnCalculator.classList.remove('active');
            navBtnAnalytics.classList.add('active');
            tabPanelCalculator.classList.remove('active');
            tabPanelAnalytics.classList.add('active');
            
            // Lazy load analytics if not cached
            if (!metadataCached) {
                fetchAnalyticsMetadata();
            }
        }
    }

    navBtnCalculator.addEventListener('click', () => switchTab('calculator'));
    navBtnAnalytics.addEventListener('click', () => switchTab('analytics'));

    // ----------------------------------------------------
    // Sliders Live Updates
    // ----------------------------------------------------
    sliders.forEach(slider => {
        const inputEl = document.getElementById(slider.id);
        const valEl = document.getElementById(slider.valId);
        
        inputEl.addEventListener('input', (e) => {
            valEl.textContent = e.target.value;
        });
    });

    // ----------------------------------------------------
    // Load Demo Record
    // ----------------------------------------------------
    btnDemo.addEventListener('click', () => {
        // Set slider inputs
        document.getElementById('input-age').value = demoPatient.Age;
        document.getElementById('val-age').textContent = demoPatient.Age;

        document.getElementById('input-bp').value = demoPatient.RestingBP;
        document.getElementById('val-bp').textContent = demoPatient.RestingBP;

        document.getElementById('input-chol').value = demoPatient.Cholesterol;
        document.getElementById('val-chol').textContent = demoPatient.Cholesterol;

        document.getElementById('input-maxhr').value = demoPatient.MaxHR;
        document.getElementById('val-maxhr').textContent = demoPatient.MaxHR;

        document.getElementById('input-oldpeak').value = demoPatient.Oldpeak;
        document.getElementById('val-oldpeak').textContent = demoPatient.Oldpeak;

        // Set radio buttons
        setRadioValue('Sex', demoPatient.Sex);
        setRadioValue('FastingBS', demoPatient.FastingBS);
        setRadioValue('ExerciseAngina', demoPatient.ExerciseAngina);

        // Set dropdown selections
        document.getElementById('input-chestpain').value = demoPatient.ChestPainType;
        document.getElementById('input-ecg').value = demoPatient.RestingECG;
        document.getElementById('input-slope').value = demoPatient.ST_Slope;
        
        // Pulse styling to indicate loading
        const formCard = document.querySelector('.form-card');
        formCard.style.transform = 'scale(0.98)';
        setTimeout(() => formCard.style.transform = 'none', 200);
    });

    function setRadioValue(name, value) {
        const radios = document.getElementsByName(name);
        for (let radio of radios) {
            if (radio.value == value) {
                radio.checked = true;
                break;
            }
        }
    }

    // ----------------------------------------------------
    // Risk Calculator Submission
    // ----------------------------------------------------
    riskForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Gather data
        const formData = new FormData(riskForm);
        const patientData = {
            Age: parseInt(formData.get('Age')),
            Sex: formData.get('Sex'),
            ChestPainType: formData.get('ChestPainType'),
            RestingBP: parseInt(formData.get('RestingBP')),
            Cholesterol: parseInt(formData.get('Cholesterol')),
            FastingBS: parseInt(formData.get('FastingBS')),
            RestingECG: formData.get('RestingECG'),
            MaxHR: parseInt(formData.get('MaxHR')),
            ExerciseAngina: formData.get('ExerciseAngina'),
            Oldpeak: parseFloat(formData.get('Oldpeak')),
            ST_Slope: formData.get('ST_Slope')
        };

        // Disable submit button during call
        const submitBtn = document.getElementById('btn-submit');
        const originalBtnText = submitBtn.textContent;
        submitBtn.disabled = true;
        submitBtn.textContent = 'Computing Risk...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientData)
            });

            const data = await response.json();
            if (data.status === 'success') {
                updatePredictionUI(data.prediction);
            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (err) {
            console.error(err);
            alert('A network error occurred. Please verify backend is running.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = originalBtnText;
        }
    });

    // Update risk UI elements
    function updatePredictionUI(pred) {
        const pct = pred.risk_percentage;
        const score = pred.risk_score;
        const band = pred.risk_band;
        
        // 1. Update text values
        resPercentage.textContent = `${pct.toFixed(1)}%`;
        resBand.textContent = `${band} Risk`;
        
        // Reset classes
        resPercentage.className = 'percentage-value';
        resBand.className = 'risk-badge';
        
        // 2. Add color coded statuses
        let statusClass = 'status-low';
        let strokeColor = 'var(--color-low)';
        if (score >= 0.25 && score < 0.50) {
            statusClass = 'status-moderate';
            strokeColor = 'var(--color-moderate)';
        } else if (score >= 0.50 && score < 0.75) {
            statusClass = 'status-high';
            strokeColor = 'var(--color-high)';
        } else if (score >= 0.75) {
            statusClass = 'status-very-high';
            strokeColor = 'var(--color-very-high)';
        }
        
        resPercentage.classList.add(statusClass);
        resBand.classList.add(statusClass);

        // 3. Animate Gauge Arc Fill
        // Circumference of semi-circle (radius 40): Math.PI * 40 = ~125.6
        // Dashoffset: 125.6 is empty, 0 is full
        const maxCircumference = 125.6;
        const dashoffset = maxCircumference * (1 - score);
        gaugeFillArc.style.strokeDashoffset = dashoffset;
        gaugeFillArc.style.stroke = strokeColor;
        
        // 4. Animate Gauge Needle Rotation
        // Score: 0 -> -90 deg, Score: 1 -> +90 deg. (total range: 180 degrees)
        const angle = (score * 180) - 90;
        gaugeNeedle.style.transform = `translate(50px,50px) rotate(${angle}deg)`;
        
        // If Very High risk, add pulsating alert shadow
        const scoreCard = document.querySelector('.result-score-card');
        if (band === 'Very High') {
            scoreCard.style.animation = 'pulseGlow 2s infinite';
        } else {
            scoreCard.style.animation = 'none';
        }

        // 5. Render Personal Drivers
        driversList.innerHTML = '';
        if (pred.top_personal_drivers && pred.top_personal_drivers.length > 0) {
            pred.top_personal_drivers.forEach(driver => {
                const row = document.createElement('div');
                row.className = 'driver-row';
                
                const label = featureLabels[driver.feature] || driver.feature;
                const isIncreasing = driver.direction === 'increases risk';
                const badgeClass = isIncreasing ? 'inc' : 'dec';
                const badgeText = isIncreasing ? 'Increases Risk ⚠️' : 'Lowers Risk ✅';
                
                // Retrieve custom readable clinical driver text description
                const textObj = driverDescriptions[driver.feature] || {};
                const explanation = textObj[driver.direction] || `This feature has a local log-odds effect size of ${driver.local_effect.toFixed(2)} on this patient's forecast.`;
                
                row.innerHTML = `
                    <div class="driver-info">
                        <span class="driver-name">${label}</span>
                        <span class="driver-desc">${explanation}</span>
                    </div>
                    <span class="driver-badge ${badgeClass}">${badgeText}</span>
                `;
                driversList.appendChild(row);
            });
        } else {
            driversList.innerHTML = '<div class="placeholder-message"><p>No drivers calculated.</p></div>';
        }

        // 6. Render Ensemble Breakdown
        modelsList.innerHTML = '';
        const order = Object.keys(pred.per_model_probability);
        order.forEach(name => {
            const prob = pred.per_model_probability[name];
            const weight = pred.model_weights_used[name];
            const row = document.createElement('div');
            row.className = 'model-bar-row';
            
            row.innerHTML = `
                <div class="model-info-row">
                    <span class="model-name">${name} <span class="model-weight">(Weight: ${(weight * 100).toFixed(0)}%)</span></span>
                    <span class="model-prob">${(prob * 100).toFixed(1)}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: ${prob * 100}%"></div>
                </div>
            `;
            modelsList.appendChild(row);
        });
    }

    // ----------------------------------------------------
    // Analytics Tab Fetching & Rendering
    // ----------------------------------------------------
    async function fetchAnalyticsMetadata() {
        try {
            const response = await fetch('/api/metadata');
            const data = await response.json();
            
            if (data.status === 'success') {
                metadataCached = data.metadata;
                renderAnalytics(metadataCached);
            } else {
                statBestModel.textContent = 'Error';
                statBestModelWeight.textContent = data.message;
            }
        } catch (err) {
            console.error(err);
            statBestModel.textContent = 'Offline';
            statBestModelWeight.textContent = 'Backend unreachable';
        }
    }

    function renderAnalytics(meta) {
        // 1. Friedman ANOVA
        const pVal = meta.friedman_p;
        statFriedmanP.textContent = `p = ${pVal.toFixed(4)}`;
        if (pVal < 0.05) {
            statFriedmanOutcome.innerHTML = 'H₀ rejected: At least one model performs significantly differently <span class="status-low">(validated)</span>.';
        } else {
            statFriedmanOutcome.innerHTML = 'H₀ kept: No significant performance difference between models overall.';
        }

        // 2. Best Model & Weights
        // Find best model by highest weight
        let bestModelName = '';
        let maxWeight = -1;
        for (let name in meta.model_weights) {
            if (meta.model_weights[name] > maxWeight) {
                maxWeight = meta.model_weights[name];
                bestModelName = name;
            }
        }
        statBestModel.textContent = bestModelName;
        statBestModelWeight.textContent = `Ensemble weight: ${(maxWeight * 100).toFixed(1)}%`;

        // 3. Render Metric bars comparison on test set
        modelComparisonBars.innerHTML = '';
        const metrics = ['accuracy', 'recall', 'f1'];
        const labels = { 'accuracy': 'Accuracy', 'recall': 'Recall', 'f1': 'F1 Score' };
        const classes = { 'accuracy': 'acc', 'recall': 'rec', 'f1': 'f1' };
        
        metrics.forEach(metric => {
            const group = document.createElement('div');
            group.className = 'model-metric-metric-group';
            group.innerHTML = `<span class="model-metric-title">${labels[metric]}</span>`;
            
            const barsRow = document.createElement('div');
            barsRow.className = 'metric-bars-row';
            
            for (let modelName in meta.test_metrics) {
                const val = meta.test_metrics[modelName][metric];
                const item = document.createElement('div');
                item.className = 'metric-bar-item';
                item.innerHTML = `
                    <span class="metric-bar-label">${modelName}</span>
                    <div class="metric-bar-track">
                        <div class="metric-bar-fill ${classes[metric]}" style="width: ${val * 100}%"></div>
                    </div>
                    <span class="metric-bar-value">${val.toFixed(2)}</span>
                `;
                barsRow.appendChild(item);
            }
            group.appendChild(barsRow);
            modelComparisonBars.appendChild(group);
        });

        // 4. Render Unified Feature Importance Chart (Chart.js)
        renderImportanceChart(meta.unified_importance);

        // 5. Render Calibration Chart (Chart.js)
        renderCalibrationChart(meta.calibration, meta.test_metrics);

        // 6. Render model strength weights
        renderModelWeights(meta.model_weights);

        // 7. Render Holm post-hoc table
        renderHolmTable(meta.holm_posthoc);

        // 8. Render correlation heatmap
        renderHeatmapGrid(meta.correlation);
    }

    function renderModelWeights(weights) {
        modelWeightsList.innerHTML = '';
        const sorted = Object.entries(weights).sort((a, b) => b[1] - a[1]);
        sorted.forEach(([name, w]) => {
            const row = document.createElement('div');
            row.className = 'weight-row';
            row.innerHTML = `
                <div class="weight-info">
                    <span class="weight-name">${name}</span>
                    <span class="weight-pct">${(w * 100).toFixed(1)}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-bar" style="width: ${w * 100}%"></div>
                </div>
            `;
            modelWeightsList.appendChild(row);
        });
    }

    function renderHolmTable(rows) {
        if (!rows || rows.length === 0) {
            holmTable.innerHTML = '<p class="loading-placeholder">No post-hoc data.</p>';
            return;
        }
        const table = document.createElement('table');
        table.className = 'holm-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Model A</th>
                    <th>Model B</th>
                    <th>Holm p</th>
                    <th>Significant</th>
                </tr>
            </thead>
            <tbody></tbody>
        `;
        const tbody = table.querySelector('tbody');
        rows.forEach(row => {
            const tr = document.createElement('tr');
            const sigClass = row.significant ? 'sig-yes' : 'sig-no';
            const sigLabel = row.significant ? 'Yes' : 'No';
            tr.innerHTML = `
                <td>${row.model_a}</td>
                <td>${row.model_b}</td>
                <td>${row.holm_p.toExponential(2)}</td>
                <td class="${sigClass}">${sigLabel}</td>
            `;
            tbody.appendChild(tr);
        });
        holmTable.innerHTML = '';
        holmTable.appendChild(table);
    }

    // Horizontal Bar Chart
    function renderImportanceChart(importance) {
        const sorted = Object.entries(importance).sort((a, b) => b[1] - a[1]);
        const labels = sorted.map(item => {
            const rawLabel = featureLabels[item[0]] || item[0];
            return rawLabel.length > 28 ? `${rawLabel.slice(0, 25)}…` : rawLabel;
        });
        const dataValues = sorted.map(item => item[1]);

        const canvas = document.getElementById('chart-feature-importance');
        const chartContainer = canvas.parentElement;
        const wrapHeight = Math.max(320, labels.length * 24 + 80);
        chartContainer.style.height = `${wrapHeight}px`;
        chartContainer.style.minHeight = `${wrapHeight}px`;

        const ctx = canvas.getContext('2d');
        
        if (featureImportanceChart) {
            featureImportanceChart.destroy();
        }

        featureImportanceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Unified Weight',
                    data: dataValues,
                    backgroundColor: 'rgba(20, 241, 199, 0.75)',
                    borderColor: 'rgba(20, 241, 199, 1)',
                    borderWidth: 1,
                    borderRadius: 5,
                    maxBarThickness: 16
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 500 },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(3)}`
                        }
                    }
                },
                layout: {
                    padding: { left: 8, right: 12, top: 6, bottom: 0 }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'var(--text-secondary)', font: { family: 'Inter', size: 10 } },
                        title: { display: true, text: 'Importance', color: 'rgba(255, 255, 255, 0.72)' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: 'var(--text-primary)',
                            font: { family: 'Outfit', size: 10 },
                            autoSkip: false,
                            maxTicksLimit: labels.length
                        }
                    }
                }
            }
        });
    }

    // Reliability Calibration Plot
    function renderCalibrationChart(calibData, metrics) {
        const datasets = [];
        
        // Colors mapping
        const colors = {
            'Logistic Regression': '#42a5f5',
            'Random Forest': '#ab47bc',
            'XGBoost': '#ffa726',
            'SVM (RBF)': '#26a69a'
        };

        for (let modelName in calibData) {
            const predProbs = calibData[modelName].prob_pred;
            const trueFractions = calibData[modelName].prob_true;
            const brier = metrics[modelName].brier;
            
            // Format coordinate points for plotting
            const coordinates = predProbs.map((p, i) => ({ x: p, y: trueFractions[i] }));
            
            datasets.push({
                label: `${modelName} (Brier: ${brier.toFixed(3)})`,
                data: coordinates,
                borderColor: colors[modelName],
                backgroundColor: colors[modelName],
                fill: false,
                tension: 0.1,
                borderWidth: 2,
                pointRadius: 4
            });
        }

        // Add perfect calibration dashed line
        datasets.push({
            label: 'Perfectly Calibrated',
            data: [{x: 0, y: 0}, {x: 1, y: 1}],
            borderColor: 'rgba(255, 255, 255, 0.25)',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0,
            borderWidth: 1.5
        });

        const ctx = document.getElementById('chart-calibration').getContext('2d');
        
        if (calibrationChart) {
            calibrationChart.destroy();
        }

        calibrationChart = new Chart(ctx, {
            type: 'line',
            data: { datasets: datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: 'var(--text-secondary)', font: { family: 'Inter', size: 10 } }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'Mean Predicted Probability', color: 'var(--text-secondary)' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'Observed Fraction Positive', color: 'var(--text-secondary)' },
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: 'var(--text-muted)' }
                    }
                }
            }
        });
    }

    // Render correlation matrix as a grid of div elements with custom tooltip and bg color
    function renderHeatmapGrid(corr) {
        correlationHeatmap.innerHTML = '';
        const cols = corr.columns;
        const n = cols.length;

        // Custom short labels to fit in grid headers
        const shortLabels = cols.map(c => {
            if (c.startsWith('ChestPainType_')) return 'CP_' + c.split('_')[1];
            if (c.startsWith('RestingECG_')) return 'ECG_' + c.split('_')[1];
            if (c.startsWith('ST_Slope_')) return 'Slope_' + c.split('_')[1];
            return c;
        });

        // Set up CSS Grid Columns layout: RowHeader + Cell cols
        correlationHeatmap.style.gridTemplateColumns = `120px repeat(${n}, 32px)`;
        
        // 1. Insert corner blank cell followed by Column Headers
        const cornerCell = document.createElement('div');
        cornerCell.className = 'heatmap-label';
        correlationHeatmap.appendChild(cornerCell);

        shortLabels.forEach(label => {
            const header = document.createElement('div');
            header.className = 'heatmap-label col-label';
            header.textContent = label;
            correlationHeatmap.appendChild(header);
        });

        // 2. Build rows
        for (let i = 0; i < n; i++) {
            // Row Label
            const rowLabel = document.createElement('div');
            rowLabel.className = 'heatmap-label row-label';
            rowLabel.textContent = shortLabels[i];
            correlationHeatmap.appendChild(rowLabel);

            // Cells in Row
            for (let j = 0; j < n; j++) {
                const val = corr.values[i][j];
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                cell.textContent = val.toFixed(1);
                
                // Color interpolation: blue (-1) -> dark slate (0) -> red (+1)
                let r, g, b, alpha;
                if (val >= 0) {
                    // Positive correlation: Red
                    r = 239; g = 68; b = 68;
                    alpha = val * 0.95; // scaling max intensity
                } else {
                    // Negative correlation: Blue
                    r = 59; g = 130; b = 246;
                    alpha = Math.abs(val) * 0.95;
                }
                
                // We overlay color over dark slate slate-900: rgba(15, 23, 42)
                cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${alpha})`;

                // Add nice tooltips
                const tt = document.createElement('div');
                tt.className = 'tooltip';
                tt.innerHTML = `<strong>${featureLabels[cols[i]] || cols[i]}</strong><br>↔ <strong>${featureLabels[cols[j]] || cols[j]}</strong><br>Pearson r: <strong>${val.toFixed(3)}</strong>`;
                cell.appendChild(tt);
                
                correlationHeatmap.appendChild(cell);
            }
        }
    }
});
