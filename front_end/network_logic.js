document.addEventListener("DOMContentLoaded", () => {
    // === UI SELECTORS AND GLOBAL STATE ===
    const cyContainer = document.getElementById('cy');
    const deviceSelector = document.getElementById('device-select');
    const networkTypeSelector = document.getElementById('network-type-select');
    const weightSlider = document.getElementById('weight-threshold');
    const thresholdLabel = document.getElementById('threshold-value');
    const infoBox = document.getElementById('info-box');

    let cy = null;
    let selectedNodes = [];
    let currentNetworkType = networkTypeSelector.value;

    // === GRAPH LOAD AND INITIALIZATION ===
    function loadGraph(device) {
        fetch(`/graph?device=${device}`)
            .then(res => res.json())
            .then(data => {
                cy = cytoscape({
                    container: cyContainer,
                    elements: data.elements,
                    layout: { name: 'preset' },
                    style: getGraphStyle(),
                    selectionType: 'additive',   // ← important
                    boxSelectionEnabled: true    // ← enables drag selection
                });

                initializeEventHandlers();
                setWeightThresholdSlider();
                applyNetworkStyle();
                applyWeightThreshold(0);
            });
    }

    // === STYLES ===
    function getGraphStyle() {
        return [
            {
                selector: 'node',
                style: {
                    'label': 'data(id)',
                    'background-color': '#00d8ff',
                    'color': '#ffffff',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'font-size': '4px',
                    'width': 'data(size)',
                    'height': 'data(size)'
                }
            },
            {
                selector: 'edge[group = "domain"]',
                style: {
                    'line-color': 'blue',
                    'width': 'data(width)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'vee',
                    'target-arrow-color': 'blue',
                    'target-arrow-scale': 0.01,
                    'opacity': 0.9
                }
            },
            {
                selector: 'edge[group = "network"]',
                style: {
                    'line-color': 'red',
                    'width': 'data(width)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'vee',
                    'target-arrow-color': 'red',
                    'target-arrow-scale': 0.01,
                    'opacity': 0.5
                }
            },
            {
                selector: 'edge.highlighted',
                style: {
                    'line-color': '#00ff00',
                    'width': 4,
                    'opacity': 1.0,
                    'z-index': 9999,
                }
            },
            {
                selector: 'node.selected',
                style: {
                    'border-width': 4,
                    'border-color': '#ffd700',
                    'selectable': true
                }
            },
            {
                selector: 'core',
                style: { 'background-color': '#000000' }
            }
        ];
    }

    // === UI / FILTER LOGIC ===
    function applyNetworkStyle() {
        if (!cy) return;
        const opacity = currentNetworkType === "domain" ? 0.0 : 0.4;
        cy.edges('[group="network"]').style({ 'opacity': opacity });
    }

    function applyWeightThreshold(threshold) {
        if (!cy) return;
        cy.edges().forEach(edge => {
            edge.style('display', edge.data('weight') >= threshold ? 'element' : 'none');
        });
    }

    function setWeightThresholdSlider() {
        let maxWeight = 0;
        cy.edges().forEach(edge => {
            const w = edge.data('weight');
            if (w > maxWeight) maxWeight = w;
        });

        weightSlider.max = Math.ceil(maxWeight);
        weightSlider.value = 0;
        thresholdLabel.textContent = "0";
    }

    // === NODE INTERACTIONS ===
    function initializeEventHandlers() {
        // Hover info
        cy.nodes().on('mouseover', evt => showNodeInfo(evt.target.animate({
                                                        style: { 'border-width': 6, 'border-color': 'hsla(188, 100%, 50%, 0.88)'}}, 
                                                        { duration: 5 }))
                                                    );
        cy.nodes().on('mouseout', () => hideInfoBox());
        
        cy.edges().on('mouseover', evt => showEdgeInfo(evt.target));
        cy.edges().on('mouseout', () => hideInfoBox());

        // Node selection for shortest path and box selection
        cy.nodes().on('tap', function (evt) {
            const event = evt.originalEvent;
            const node = evt.target;
            const id = node.id();

            // Only do shortest path if Shift key is held
            if (!event.shiftKey) return;

            if (selectedNodes.includes(id)) {
                selectedNodes = selectedNodes.filter(n => n !== id);
                node.removeClass('selected');
            } else {
                selectedNodes.push(id);
                node.addClass('selected');
            }

            if (selectedNodes.length === 2) {
                const [src, dst] = selectedNodes;
                selectedNodes.forEach(n => cy.getElementById(n).removeClass('selected'));
                selectedNodes = [];
                getShortestPaths(src, dst);
            }
        });

        cy.on('select', 'node', function () {
            const selected = cy.nodes(':selected');
            const summaries = selected.map(n => {
                const d = n.data();
                return `
                    <strong>${d.id}</strong><br>
                    <br>
                    Time spent in seconds:<br>
                    AvgAbs: ${d.AvgAbs?.toFixed(2) || 'n/a'}s<br>
                    Total: ${d.Total?.toFixed(2) || 'n/a'}s<br>
                    AvgRel: ${d.AvgRel?.toFixed(2) || 'n/a'}s<br>
                    <br>
                    Flow: ${d.flow?.toFixed(2) || 'n/a'}<br>
                `;
            });

            infoBox.innerHTML = `
                <strong>Selected ${selected.length} node(s)</strong><br><br>
                ${summaries.join('<hr>')}
            `;
            infoBox.style.display = 'block';
        });


    }

    function handleNodeSelection(node) {
        const id = node.id();

        if (selectedNodes.includes(id)) {
            selectedNodes = selectedNodes.filter(n => n !== id);
            node.removeClass('selected');
        } else {
            selectedNodes.push(id);
            node.addClass('selected');
        }

        if (selectedNodes.length === 2) {
            const [src, dst] = selectedNodes;
            selectedNodes.forEach(n => cy.getElementById(n).removeClass('selected'));
            selectedNodes = [];
            getShortestPaths(src, dst);
        }
    }

    // === PATH FINDING ===
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

                infoBox.innerHTML = `
                    <strong>${data.paths.length} shortest paths from ${source} → ${target}</strong><br><br>
                    ${data.paths.map((p, i) =>
                        `<strong>Path ${i + 1}:</strong> ${p.path.join(' → ')}<br>Score: ${p.score_abs.toFixed(2)}`
                    ).join('<br><br>')}
                `;
                infoBox.style.display = 'block';
            })
            .catch(err => {
                console.error("Error fetching shortest path:", err);
                alert("Failed to fetch shortest path.");
            });
    }

    // === INFO PANEL ===
    function showNodeInfo(node) {
        const data = node.data();
        infoBox.innerHTML = `
            <strong>Node: ${data.id}</strong><br>
            <br>    
            AvgAbs: ${data.AvgAbs?.toFixed(2) || 'n/a'}s<br>
            Total: ${data.Total?.toFixed(2) || 'n/a'}s<br>
            AvgRel: ${data.AvgRel?.toFixed(2) || 'n/a'}s<br>
            <br>
            Flow: ${data.flow?.toFixed(2) || 'n/a'}<br> 
        `;
        infoBox.style.display = 'block';
    }

    function showEdgeInfo(edge) {
        const data = edge.data();
        infoBox.innerHTML = `
            <strong>Edge: ${data.source} → ${data.target}</strong><br>
            Weight: ${data.weight || 'n/a'}
        `;
        infoBox.style.display = 'block';
    }

    function hideInfoBox() {
        infoBox.style.display = 'none';
    }

    // === CONTROL EVENTS ===
    deviceSelector.addEventListener('change', () => loadGraph(deviceSelector.value));

    networkTypeSelector.addEventListener('change', () => {
        currentNetworkType = networkTypeSelector.value;
        applyNetworkStyle();
    });

    weightSlider.addEventListener('input', () => {
        const threshold = parseInt(weightSlider.value);
        thresholdLabel.textContent = threshold;
        applyWeightThreshold(threshold);
    });

    document.getElementById('clear-selection').addEventListener('click', () => {
    cy.nodes().unselect();
    cy.edges().removeClass('highlighted');
    infoBox.style.display = 'none';
    });


    // === INITIAL LOAD ===
    loadGraph(deviceSelector.value);
});
