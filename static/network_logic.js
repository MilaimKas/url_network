document.querySelector('#info-box .content').innerHTML = "Shift+click two nodes to compute shortest paths. Results appear here.";
document.querySelector('#hover-box .content').innerHTML = "Hover nodes or edges to see details.";
document.addEventListener("DOMContentLoaded", () => {

    const cyContainer = document.getElementById('cy');
    const deviceSelector = document.getElementById('device-select');
    const networkTypeSelector = document.getElementById('network-type-select');
    const infoBox = document.getElementById('info-box');
    const sliderElem = document.getElementById('weight-range-slider');

    let cy = null;
    let selectedNodes = [];
    let currentNetworkType = networkTypeSelector.value;
    let dateList = [];
    let currentDateIndex = -1;
    let globalMaxWeight = null;
    let allMetricsCache = null;


    // === TABLE INIT ===
    const dataTable = $('#summary-table').DataTable({
        paging: false,
        searching: false,
        info: false,
        ordering: true,
        columns: [
            { title: "Node" },
            { title: "AvgAbs" },
            { title: "Total" },
            { title: "AvgRel" },
            { title: "Flow" }
        ]
    });

    // === Get all time period as start date ===
    fetch("/available-dates")
        .then(res => res.json())
        .then(data => {
            dateList = data;

            const slider = document.getElementById("date-slider");
            const totalOptions = dateList.length + 1;  // +1 for "All time"

            slider.max = totalOptions - 1;
            slider.value = totalOptions - 1;

            document.getElementById("date-display").textContent = "All time";
        });


    // === GRAPH LOAD ===
    function loadGraph(device, date = null) {
        let url = `/graph?device=${device}`;
        if (date && date !== "All time") {
            url += `&date=${date}`;
        }

        fetch(url)
            .then(res => res.json())
            .then(data => {
                if (cy) {
                    cy.destroy(); // 🔥 Prevent multiple instances
                }

                cy = cytoscape({
                    container: cyContainer,
                    elements: data.elements,
                    layout: { name: 'preset' },
                    style: getGraphStyle(),
                    selectionType: 'additive',
                    boxSelectionEnabled: true
                });

                initializeEventHandlers();
                if (date === null) {
                setWeightThresholdSlider();  // only recompute on "All time"
                }                
                applyNetworkStyle();     // ✅ important

                if (sliderElem.noUiSlider) {
                    const [min, max] = sliderElem.noUiSlider.get().map(v => Math.floor(v));
                    applyWeightThresholdRange(min, max);
                }
        });

        fetchAllMetricsOnce().then(() => {
            plotMetric('entropy');
        });

    }

    // === STYLE SETUP ===
    function getGraphStyle() {
        return [
            {
            selector: 'node',
                style: {
                    'label': 'data(id)',
                    'background-color': 'rgba(0, 216, 255, 1)',
                    'color': 'rgba(255, 255, 255, 1)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '4px',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'selectable': true
                }
            },
            {
            selector: 'edge[group = "domain-backbone"]',
                style: {
                    'line-color': '#444',
                    'width': 0.5,
                    'curve-style': 'straight',
                    'opacity': 0.4,
                    'z-index': 1,
                    'target-arrow-shape': 'none'
                }
            },
            {
            selector: 'edge[group = "network"]',
                style: {
                    'line-color': 'data(color)',
                    'width': 'data(width)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'target-arrow-color': 'red',
                    'target-arrow-fill': 'hollow',
                    'arrow-scale': 0.2
                }
            },
            {
            selector: 'edge.highlighted',
                style: {
                    'line-color': 'rgba(0, 255, 0, 1)',
                    'width': 4,
                    'opacity': 1.0,
                    'z-index': 9999
                }
            },
            {
            selector: 'node.hovered',
                style: {
                    'border-width': 6,
                    'border-color': 'rgba(52, 115, 70, 1)'
                }
            },
            {
            selector: 'node:selected',
                style: {
                    'border-width': 6,
                    'border-color': 'rgba(237, 203, 6, 0.6)'
                }
            },
            {
            selector: 'core',
                style: { 'background-color': '#000000' }
            }
        ];
    }

    // control edge visibility based on weight threshold range    
    function applyWeightThresholdRange(min, max) {
        if (!cy) return;

        cy.edges().forEach(edge => {
            const group = edge.data('group');
            const weight = edge.data('weight') || 0;

            if (group === "network") {
                const visible = weight >= min && weight <= max;
                edge.style('display', visible ? 'element' : 'none');
            } else {
                edge.style('display', 'element');  // domain-backbone always visible
            }
        });
    }

    // set slider max/min based on max edge weight
    function setWeightThresholdSlider() {
        if (globalMaxWeight === null) {
            let maxWeight = 0;
            cy.edges('[group="network"]').forEach(edge => {
                const w = edge.data('weight');
                if (w > maxWeight) maxWeight = w;
            });

            globalMaxWeight = Math.ceil(maxWeight);
        }

        const sliderElem = document.getElementById('weight-range-slider');

        if (!sliderElem.noUiSlider) {
            noUiSlider.create(sliderElem, {
                start: [0, globalMaxWeight],
                connect: true,
                range: {
                    min: 0,
                    max: globalMaxWeight
                },
                step: 1,
                tooltips: true
            });

            sliderElem.noUiSlider.on('update', function ([min, max]) {
                document.getElementById('range-min').textContent = Math.floor(min);
                document.getElementById('range-max').textContent = Math.ceil(max);
                applyWeightThresholdRange(Math.floor(min), Math.ceil(max));
            });
        }
    }

    // update summary table with selected nodes
    function updateSummaryTable(nodes) {
        dataTable.clear();

        nodes.forEach(node => {
            const d = node.data();
            dataTable.row.add([
                d.id,
                d.AvgAbs?.toFixed(2) || 'n/a',
                d.Total?.toFixed(2) || 'n/a',
                d.AvgRel?.toFixed(2) || 'n/a',
                d.flow?.toFixed(2) || 'n/a'
            ]);
        });

        dataTable.draw();
    }

    // === CY EVENT HANDLERS ===
    function initializeEventHandlers() {

        // Node hover effect
        cy.nodes().on('mouseover', evt => {
            evt.target.addClass('hovered');
            showNodeInfo(evt.target);
            // show info panel (subtle)
            document.getElementById('hover-box').style.boxShadow = '0 8px 20px rgba(0,0,0,0.6)';
        });

        cy.nodes().on('mouseout', evt => {
            evt.target.removeClass('hovered');
            hideInfoBox();
            document.getElementById('hover-box').style.boxShadow = 'none';
        });

        // Edge hover → info panel
        cy.edges().on('mouseover', evt => showEdgeInfo(evt.target));
        cy.edges().on('mouseout', () => hideInfoBox());

        // Shift+click = shortest path trigger
        cy.nodes().on('tap', evt => {
            const node = evt.target;
            const id = node.id();
            const isShift = evt.originalEvent.shiftKey;

            if (!isShift) return;

            if (selectedNodes.includes(id)) {
                selectedNodes = selectedNodes.filter(n => n !== id);
                node.unselect();
            } else {
                selectedNodes.push(id);
                node.select();    
            }

            if (selectedNodes.length === 2) {
                const [src, dst] = selectedNodes;
                selectedNodes.forEach(n => cy.getElementById(n).unselect());
                selectedNodes = [];
                getShortestPaths(src, dst);
            }
        });


        // Box selection → update table
        cy.on('select', 'node', () => {
            updateSummaryTable(cy.nodes(':selected'));
        });

        cy.on('unselect', 'node', () => {
            updateSummaryTable(cy.nodes(':selected'));
        });
    }

    function getShortestPaths(source, target) {
        const params = new URLSearchParams({
            src: source,
            dst: target,
            device: deviceSelector.value,
            max_paths: 5
        });

        fetch(`/shortest-path?${params.toString()}`)
            .then(res => res.json())
            .then(data => {
                cy.edges().removeClass('highlighted');
                if (!data.paths || data.paths.length === 0) {
                    alert(data.message || "No paths found.");
                    return;
                }

                const allPathEdges = new Set();
                data.paths.forEach(p => {
                    for (let i = 0; i < p.path.length - 1; i++) {
                        const u = p.path[i];
                        const v = p.path[i + 1];
                        const edge = cy.edges().filter(e => e.source().id() === u && e.target().id() === v);
                        edge.forEach(e => allPathEdges.add(e));
                    }
                });
                allPathEdges.forEach(e => e.addClass('highlighted'));

                document.querySelector('#info-box .content').innerHTML = `
                <strong>${data.paths.length} shortest paths from ${source} → ${target}</strong><br><br>
                ${data.paths.map((p, i) =>
                    `<strong>Path ${i + 1}:</strong> ${p.path.join(' → ')}<br>Score: ${p.score_abs.toFixed(2)}`
                ).join('<br><br>')}
                `;

            })
            .catch(err => {
                console.error("Error fetching shortest path:", err);
                alert("Failed to fetch shortest path.");
            });
    }

    function applyNetworkStyle() {
        if (!cy) return;
        const opacity = currentNetworkType === "domain" ? 0.0 : 0.4;
        cy.edges('[group="network"]').style({ 'opacity': opacity });
    }

    function showEdgeInfo(edge) {
        const data = edge.data();
        document.querySelector('#hover-box .content').innerHTML = `
            <strong>Edge: ${data.source} → ${data.target}</strong><br>
            Weight: ${data.weight || 'n/a'}
        `;
    }

    function showNodeInfo(node) {
        const data = node.data();
        document.querySelector('#hover-box .content').innerHTML = `
            <strong>Node: ${data.id}</strong><br>
            <br>    
            AvgAbs: ${data.AvgAbs?.toFixed(2) || 'n/a'}s<br>
            AvgRel: ${data.AvgRel?.toFixed(2) || 'n/a'}s<br>
            Total: ${data.Total?.toFixed(2) || 'n/a'}s<br>
            <br>
            Flow: ${data.flow?.toFixed(2) || 'n/a'}<br> 
        `;
    }

    function hideInfoBox() {
        document.querySelector('#hover-box .content').innerHTML = "";
    }

    function fetchAllMetricsOnce() {
        return fetch(`/graph-metrics?device=${deviceSelector.value}`)
            .then(res => res.json())
            .then(data => {
            allMetricsCache = data;
            return data;
            });
    }

    function plotMetric(metric) {
        if (!allMetricsCache || !allMetricsCache[metric]) return;

        const data = allMetricsCache[metric];
        const ctx = document.getElementById('metric-chart').getContext('2d');
        const labels = data.map(d => d.date);
        const values = data.map(d => d.value);

        if (window.metricChart) {
            window.metricChart.destroy();
        }

        window.metricChart = new Chart(ctx, {
            type: 'line',
            data: {
            labels: labels,
            datasets: [{
                label: metric,
                data: values,
                borderColor: 'rgba(255, 204, 0, 0.9)',
                backgroundColor: 'rgba(255, 204, 0, 0.1)',
                fill: true,
                tension: 0.3,
                pointRadius: 2
            }]
            },
            options: {
            scales: {
                x: {
                title: { display: true, text: "Date" },
                ticks: { maxRotation: 45, minRotation: 45 }
                },
                y: {
                title: { display: true, text: metric },
                beginAtZero: false
                }
            },
            plugins: {
                legend: { display: false }
            }
            }
        });
    }

   
    // === CONTROL EVENTS ===
    deviceSelector.addEventListener('change', () => loadGraph(deviceSelector.value));
    networkTypeSelector.addEventListener('change', () => {
        currentNetworkType = networkTypeSelector.value;
        applyNetworkStyle();
    });

    document.getElementById('metric-select').addEventListener('change', (e) => {
        const selectedMetric = e.target.value;
        plotMetric(selectedMetric);
    });

    document.getElementById('clear-selection').addEventListener('click', () => {
        cy.nodes().unselect();
        cy.edges().removeClass('highlighted');
        infoBox.style.display = 'none';
        dataTable.clear().draw();
    });

    document.getElementById("date-slider").addEventListener("input", (e) => {
        const index = parseInt(e.target.value);
        currentDateIndex = index;

        const dateDisplay = document.getElementById("date-display");

        if (index === dateList.length) {
            dateDisplay.textContent = "All time";
            loadGraph(deviceSelector.value, null);
        } else {
            const selected = dateList[index];
            dateDisplay.textContent = selected;
            loadGraph(deviceSelector.value, selected);
        }
    });




    // === INITIALIZE ===
    loadGraph(deviceSelector.value);
    plotMetric("entropy")
});
