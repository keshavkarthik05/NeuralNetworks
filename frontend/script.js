const API = "";


/* =========================================================
   GLOBAL CHARTS
========================================================= */

let shapChart = null;
let riskChart = null;
let moduleChart = null;


/* =========================================================
   NAVIGATION
========================================================= */

function showPage(pageId, button) {

    document
        .querySelectorAll(".page")
        .forEach(page => {

            page.classList.remove(
                "active-page"
            );

        });


    document
        .getElementById(pageId)
        .classList.add(
            "active-page"
        );


    document
        .querySelectorAll(".nav-item")
        .forEach(item => {

            item.classList.remove(
                "active"
            );

        });


    button.classList.add(
        "active"
    );


    if (pageId === "analytics") {

        loadCohort();

    }


    if (pageId === "model") {

        loadMetrics();

    }

}


/* =========================================================
   LOAD STUDENTS
========================================================= */

async function loadStudents() {

    try {

        const response =
            await fetch(
                `${API}/api/students`
            );


        const students =
            await response.json();


        const select =
            document.getElementById(
                "studentSelect"
            );


        select.innerHTML = "";


        students.forEach(
            student => {

                const option =
                    document.createElement(
                        "option"
                    );


                option.value =
                    student.student_id;


                option.textContent =
                    `${student.student_id}
                    — ${student.module}
                    / ${student.presentation}`;


                select.appendChild(
                    option
                );

            }
        );


    } catch (error) {

        console.error(error);

        alert(
            "Could not connect to backend."
        );

    }

}


/* =========================================================
   PREDICT STUDENT
========================================================= */

async function predictStudent() {

    const select =
        document.getElementById(
            "studentSelect"
        );


    const studentId =
        Number(select.value);


    if (!studentId) {

        alert(
            "Please select a student."
        );

        return;

    }


    try {

        const response =
            await fetch(
                `${API}/api/predict`,
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        student_id:
                            studentId

                    })

                }
            );


        if (!response.ok) {

            throw new Error(
                "Prediction failed"
            );

        }


        const data =
            await response.json();


        displayPrediction(
            data
        );


    } catch (error) {

        console.error(error);

        alert(
            "Could not generate prediction."
        );

    }

}


/* =========================================================
   DISPLAY PREDICTION
========================================================= */

function displayPrediction(data) {


    document
        .getElementById(
            "predictionResult"
        )
        .style.display = "block";


    const prediction =
        data.prediction;


    const student =
        data.student;


    const behaviour =
        data.behaviour;


    /* Risk */

    const riskText =
        document.getElementById(
            "riskText"
        );


    riskText.textContent =
        riskText.textContent =
        prediction.risk;


    const probability =
        document.getElementById(
            "probabilityText"
        );


    probability.textContent =
        `${prediction.percentage.toFixed(1)}%`;

    const interpretation =
    document.getElementById(
        "riskInterpretation"
    );


if (prediction.risk === "HIGH") {

    interpretation.innerHTML = `
        <strong>Attention recommended</strong>
        <p>
            The model identifies this student as having a
            relatively high predicted dropout risk based on
            their observed learning behaviour during the
            first eight weeks.
        </p>
    `;

}
else if (prediction.risk === "MEDIUM") {

    interpretation.innerHTML = `
        <strong>Monitoring recommended</strong>
        <p>
            The model identifies moderate predicted dropout
            risk. Continued monitoring of engagement,
            assessments and learning activity may be useful.
        </p>
    `;

}
else {

    interpretation.innerHTML = `
        <strong>Lower predicted risk</strong>
        <p>
            The model estimates a relatively low dropout risk
            based on the student's observed learning behaviour.
        </p>
    `;

}


    /* Risk styling */

    const banner =
        document.querySelector(
            ".risk-banner"
        );


    if (prediction.risk === "HIGH") {

        banner.style.borderLeft =
            "6px solid #d92d20";

        riskText.style.color =
            "#d92d20";

    }

    else if (
        prediction.risk === "MEDIUM"
    ) {

        banner.style.borderLeft =
            "6px solid #f79009";

        riskText.style.color =
            "#f79009";

    }

    else {

        banner.style.borderLeft =
            "6px solid #12b76a";

        riskText.style.color =
            "#12b76a";

    }


    /* Student */

    document.getElementById(
        "studentId"
    ).textContent =
        student.id;


    document.getElementById(
        "module"
    ).textContent =
        student.module;


    document.getElementById(
        "presentation"
    ).textContent =
        student.presentation;


    document.getElementById(
        "ageBand"
    ).textContent =
        student.age_band;


    /* Behaviour */

    document.getElementById(
        "clicks"
    ).textContent =
        formatNumber(
            behaviour.clicks
        );


    document.getElementById(
        "activeDays"
    ).textContent =
        behaviour.active_days;


    document.getElementById(
        "activeWeeks"
    ).textContent =
        behaviour.active_weeks;


    document.getElementById(
        "uniqueSites"
    ).textContent =
        behaviour.unique_sites;


    document.getElementById(
        "assessments"
    ).textContent =
        behaviour.assessments;


    document.getElementById(
        "meanScore"
    ).textContent =
        behaviour.mean_score === null
            ? "N/A"
            : behaviour.mean_score.toFixed(1);


    document.getElementById(
        "lateSubmissions"
    ).textContent =
        behaviour.late_submissions;


    document.getElementById(
        "engagementChange"
    ).textContent =
        behaviour.engagement_change;


    /* SHAP */

    createShapChart(
        data.shap.top
    );


    createFactors(
        data.shap.positive,
        data.shap.negative
    );

}


/* =========================================================
   SHAP CHART
========================================================= */

function createShapChart(values) {


    const sorted =
        [...values].sort(
            (a, b) =>
                a.value - b.value
        );


    const labels =
        sorted.map(
            x => x.name
        );


    const data =
        sorted.map(
            x => x.value
        );


    const backgroundColors =
        data.map(
            value =>
                value >= 0
                    ? "#ef4444"
                    : "#22c55e"
        );


    const ctx =
        document
            .getElementById(
                "shapChart"
            )
            .getContext("2d");


    if (shapChart) {

        shapChart.destroy();

    }


    shapChart =
        new Chart(
            ctx,
            {

                type: "bar",

                data: {

                    labels: labels,

                    datasets: [

                        {

                            label:
                                "SHAP Contribution",

                            data: data,

                            backgroundColor:
                                backgroundColors,

                            borderRadius: 5

                        }

                    ]

                },

                options: {

                    indexAxis: "y",

                    responsive: true,

                    plugins: {

                        legend: {
                            display: false
                        }

                    },

                    scales: {

                        x: {

                            title: {

                                display: true,

                                text:
                                    "SHAP Value"

                            }

                        }

                    }

                }

            }
        );

}


/* =========================================================
   FACTORS
========================================================= */

function createFactors(
    positive,
    negative
) {


    const positiveContainer =
        document.getElementById(
            "positiveFactors"
        );


    const negativeContainer =
        document.getElementById(
            "negativeFactors"
        );


    positiveContainer.innerHTML = "";

    negativeContainer.innerHTML = "";


    positive.forEach(
        factor => {

            positiveContainer.innerHTML += `

                <div class="factor-item">

                    <span>
                        ${factor.name}
                    </span>

                    <span class="factor-value"
                          style="color:#d92d20">

                        +${factor.value.toFixed(3)}

                    </span>

                </div>

            `;

        }
    );


    negative.forEach(
        factor => {

            negativeContainer.innerHTML += `

                <div class="factor-item">

                    <span>
                        ${factor.name}
                    </span>

                    <span class="factor-value"
                          style="color:#12b76a">

                        ${factor.value.toFixed(3)}

                    </span>

                </div>

            `;

        }
    );

}


/* =========================================================
   COHORT
========================================================= */

/* =========================================================
   COHORT ANALYTICS
========================================================= */

async function loadCohort() {

    try {

        const response =
            await fetch(
                `${API}/api/cohort`
            );


        if (!response.ok) {

            throw new Error(
                "Cohort request failed"
            );

        }


        const data =
            await response.json();


        /* -----------------------------------------------
           Summary
        ------------------------------------------------ */

        document.getElementById(
            "totalStudents"
        ).textContent =
            formatNumber(
                data.total_students
            );


        document.getElementById(
            "highRisk"
        ).textContent =
            formatNumber(
                data.high_risk
            );


        document.getElementById(
            "mediumRisk"
        ).textContent =
            formatNumber(
                data.medium_risk
            );


        document.getElementById(
            "observedDropout"
        ).textContent =
            data.observed_dropout === null
                ? "N/A"
                : `${data.observed_dropout.toFixed(1)}%`;


        /* -----------------------------------------------
           Charts
        ------------------------------------------------ */

        createRiskChart(data);

        createModuleChart(
            data.modules
        );


        /* -----------------------------------------------
           Tables
        ------------------------------------------------ */

        createModuleTable(
            data.modules
        );


        createHighRiskTable(
            data.high_risk_students
        );


    } catch (error) {

        console.error(
            "Cohort error:",
            error
        );

    }

}


/* =========================================================
   RISK DISTRIBUTION
========================================================= */

function createRiskChart(data) {

    const ctx =
        document
            .getElementById(
                "riskChart"
            )
            .getContext("2d");


    if (riskChart) {

        riskChart.destroy();

    }


    riskChart =
        new Chart(
            ctx,
            {

                type: "doughnut",

                data: {

                    labels: [
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ],

                    datasets: [

                        {

                            data: [

                                data.low_risk,
                                data.medium_risk,
                                data.high_risk

                            ],

                            backgroundColor: [

                                "#22c55e",
                                "#f59e0b",
                                "#ef4444"

                            ],

                            borderWidth: 0

                        }

                    ]

                },

                options: {

                    responsive: true,

                    cutout: "68%",

                    plugins: {

                        legend: {

                            position: "bottom"

                        }

                    }

                }

            }
        );

}


/* =========================================================
   MODULE CHART
========================================================= */

function createModuleChart(
    modules
) {

    const sorted =
        [...modules].sort(
            (a, b) =>
                b.predicted_risk -
                a.predicted_risk
        );


    const labels =
        sorted.map(
            x => x.module
        );


    const predicted =
        sorted.map(
            x => x.predicted_risk
        );


    const observed =
        sorted.map(
            x => x.observed_dropout
        );


    const ctx =
        document
            .getElementById(
                "moduleChart"
            )
            .getContext("2d");


    if (moduleChart) {

        moduleChart.destroy();

    }


    moduleChart =
        new Chart(
            ctx,
            {

                type: "bar",

                data: {

                    labels: labels,

                    datasets: [

                        {

                            label:
                                "Predicted Risk",

                            data:
                                predicted,

                            backgroundColor:
                                "#2563eb",

                            borderRadius: 5

                        },

                        {

                            label:
                                "Observed Dropout",

                            data:
                                observed,

                            backgroundColor:
                                "#94a3b8",

                            borderRadius: 5

                        }

                    ]

                },

                options: {

                    responsive: true,

                    scales: {

                        y: {

                            beginAtZero: true,

                            title: {

                                display: true,

                                text:
                                    "Percentage (%)"

                            }

                        }

                    }

                }

            }
        );

}


/* =========================================================
   MODULE TABLE
========================================================= */

function createModuleTable(
    modules
) {

    const table =
        document.getElementById(
            "moduleTable"
        );


    table.innerHTML = "";


    modules
        .sort(
            (a, b) =>
                b.predicted_risk -
                a.predicted_risk
        )
        .forEach(
            module => {

                const row =
                    document.createElement(
                        "tr"
                    );


                row.innerHTML = `

                    <td>
                        <strong>
                            ${module.module}
                        </strong>
                    </td>

                    <td>
                        ${formatNumber(
                            module.students
                        )}
                    </td>

                    <td>

                        <span class="risk-pill predicted">

                            ${module.predicted_risk.toFixed(1)}%

                        </span>

                    </td>

                    <td>

                        ${
                            module.observed_dropout === null

                                ? "N/A"

                                : `<span class="risk-pill observed">
                                    ${module.observed_dropout.toFixed(1)}%
                                  </span>`
                        }

                    </td>

                `;


                table.appendChild(
                    row
                );

            }
        );

}


/* =========================================================
   HIGH-RISK STUDENT TABLE
========================================================= */

function createHighRiskTable(
    students
) {

    const table =
        document.getElementById(
            "highRiskTable"
        );


    const count =
        document.getElementById(
            "highRiskCount"
        );


    table.innerHTML = "";


    count.textContent =
        `${students.length} students shown`;


    if (students.length === 0) {

        table.innerHTML = `

            <tr>

                <td
                    colspan="7"
                    class="empty-state"
                >

                    No high-risk students found.

                </td>

            </tr>

        `;

        return;

    }


    students.forEach(
        student => {

            const row =
                document.createElement(
                    "tr"
                );


            row.innerHTML = `

                <td>

                    <strong>
                        ${student.student_id}
                    </strong>

                </td>

                <td>
                    ${student.module}
                </td>

                <td>
                    ${student.presentation}
                </td>

                <td>

                    <span class="high-risk-pill">

                        ${student.risk.toFixed(1)}%

                    </span>

                </td>

                <td>
                    ${formatNumber(
                        student.clicks
                    )}
                </td>

                <td>
                    ${student.active_days}
                </td>

                <td>

                    ${
                        student.mean_score === null
                            ? "N/A"
                            : student.mean_score.toFixed(1)
                    }

                </td>

            `;


            table.appendChild(
                row
            );

        }
    );

}


/* =========================================================
   RISK CHART
========================================================= */

function createRiskChart(data) {

    const ctx =
        document
            .getElementById(
                "riskChart"
            )
            .getContext("2d");


    if (riskChart) {

        riskChart.destroy();

    }


    riskChart =
        new Chart(
            ctx,
            {

                type: "doughnut",

                data: {

                    labels: [
                        "Low Risk",
                        "Medium Risk",
                        "High Risk"
                    ],

                    datasets: [

                        {

                            data: [

                                data.low_risk,
                                data.medium_risk,
                                data.high_risk

                            ],

                            backgroundColor: [

                                "#22c55e",
                                "#f59e0b",
                                "#ef4444"

                            ]

                        }

                    ]

                },

                options: {

                    responsive: true,

                    plugins: {

                        legend: {

                            position:
                                "bottom"

                        }

                    }

                }

            }
        );

}


/* =========================================================
   MODULE CHART
========================================================= */

function createModuleChart(
    modules
) {


    const sorted =
        [...modules].sort(
            (a, b) =>
                b.mean_risk -
                a.mean_risk
        );


    const labels =
        sorted.map(
            x => x.module
        );


    const values =
        sorted.map(
            x => x.mean_risk
        );


    const ctx =
        document
            .getElementById(
                "moduleChart"
            )
            .getContext("2d");


    if (moduleChart) {

        moduleChart.destroy();

    }


    moduleChart =
        new Chart(
            ctx,
            {

                type: "bar",

                data: {

                    labels: labels,

                    datasets: [

                        {

                            label:
                                "Average Risk (%)",

                            data: values,

                            backgroundColor:
                                "#2563eb",

                            borderRadius: 5

                        }

                    ]

                },

                options: {

                    responsive: true,

                    plugins: {

                        legend: {
                            display: false
                        }

                    },

                    scales: {

                        y: {

                            beginAtZero: true,

                            title: {

                                display: true,

                                text:
                                    "Risk (%)"

                            }

                        }

                    }

                }

            }
        );

}


/* =========================================================
   MODEL METRICS
========================================================= */

async function loadMetrics() {

    try {

        const response =
            await fetch(
                `${API}/api/metrics`
            );


        const data =
            await response.json();


        const m =
            data.metrics;


        setMetric(
            "accuracy",
            m.accuracy
        );


        setMetric(
            "precision",
            m.precision
        );


        setMetric(
            "recall",
            m.recall
        );


        setMetric(
            "f1",
            m.f1
        );


        setMetric(
            "rocauc",
            m.roc_auc
        );


        setMetric(
            "prauc",
            m.pr_auc
        );


    } catch (error) {

        console.error(error);

    }

}


/* =========================================================
   METRIC HELPER
========================================================= */

function setMetric(
    id,
    value
) {

    if (value === undefined) {

        return;

    }


    if (value <= 1) {

        value *= 100;

    }


    document.getElementById(
        id
    ).textContent =
        `${value.toFixed(2)}%`;

}


/* =========================================================
   FORMAT NUMBER
========================================================= */

function formatNumber(
    value
) {

    return Number(
        value
    ).toLocaleString();

}


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    () => {

        loadStudents();

    }
);