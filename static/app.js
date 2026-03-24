// API 基础路径
const API_BASE = '/api';

// 任务状态管理器 - 防止重复点击
const TaskManager = {
    tasks: {
        classification_train: false,
        classification_predict: false,
        classification_test: false,
        segmentation_train: false,
        segmentation_predict: false,
        segmentation_test: false
    },

    isRunning(taskName) {
        return this.tasks[taskName] === true;
    },

    start(taskName) {
        if (this.tasks[taskName]) {
            return false;
        }
        this.tasks[taskName] = true;
        this.updateButtonState(taskName, true);
        return true;
    },

    end(taskName) {
        this.tasks[taskName] = false;
        this.updateButtonState(taskName, false);
    },

    updateButtonState(taskName, isRunning) {
        const buttonMap = {
            'classification_train': 'class-train-btn',
            'classification_predict': 'class-predict-btn',
            'classification_test': 'class-test-btn',
            'segmentation_train': 'seg-train-btn',
            'segmentation_predict': 'seg-predict-btn',
            'segmentation_test': 'seg-test-btn'
        };

        const btnId = buttonMap[taskName];
        if (btnId) {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.disabled = isRunning;
                if (isRunning) {
                    btn.dataset.originalText = btn.innerHTML;
                    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 处理中...';
                } else {
                    btn.innerHTML = btn.dataset.originalText || btn.innerHTML;
                }
            }
        }
    }
};

// 页面切换
function showPage(page) {
    document.querySelectorAll('[id^="page-"]').forEach(el => el.classList.add('hidden'));
    document.getElementById(`page-${page}`).classList.remove('hidden');
    
    if (page === 'home') {
        loadSystemStatus();
    }
}

// 显示加载模态框
function showLoading(text = '处理中...') {
    document.getElementById('loadingText').textContent = text;
    const modal = new bootstrap.Modal(document.getElementById('loadingModal'));
    modal.show();
    return modal;
}

// 隐藏加载模态框
function hideLoading() {
    const modalEl = document.getElementById('loadingModal');
    const modal = bootstrap.Modal.getInstance(modalEl);

    if (modal) {
        modal.hide();

        // 等待 Bootstrap 的过渡动画完成，然后强制隐藏
        setTimeout(() => {
            if (modal._isShown) {
                forceHideModal(modalEl);
            }
        }, 500);
    } else {
        forceHideModal(modalEl);
    }
}

// 强制隐藏模态框的辅助函数
function forceHideModal(modalEl) {
    // 移除焦点，避免警告
    if (document.activeElement && modalEl.contains(document.activeElement)) {
        document.activeElement.blur();
    }

    // 销毁 Bootstrap 模态框实例
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) {
        modal.dispose();
    }

    // 移除所有模态框相关的类和属性
    modalEl.classList.remove('show');
    modalEl.classList.remove('fade');
    modalEl.style.display = 'none';
    modalEl.setAttribute('aria-hidden', 'true');
    modalEl.removeAttribute('aria-modal');

    // 移除所有遮罩层
    const backdrops = document.querySelectorAll('.modal-backdrop');
    backdrops.forEach(backdrop => backdrop.remove());

    // 恢复 body 样式
    document.body.classList.remove('modal-open');
    document.body.style.overflow = '';
    document.body.style.paddingRight = '';
}

// 加载系统状态
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/system/status`);
        const data = await response.json();
        
        const gpuStatus = data.gpu.available ? 
            `<span class="status-badge status-available"><i class="bi bi-check-circle"></i> 可用</span>` :
            `<span class="status-badge status-unavailable"><i class="bi bi-x-circle"></i> 不可用</span>`;
        
        let gpuInfo = '';
        if (data.gpu.available && data.selection) {
            const sel = data.selection;
            gpuInfo = `
                <div class="alert alert-info">
                    <strong><i class="bi bi-gpu-card"></i> 当前使用:</strong> ${sel.message}
                </div>
            `;
        }
        
        document.getElementById('system-status').innerHTML = `
            ${gpuInfo}
            <div class="row">
                <div class="col-md-6">
                    <p><strong>GPU状态:</strong> ${gpuStatus}</p>
                    <p><strong>设备数量:</strong> ${data.gpu.device_count}</p>
                    <p><strong>平台:</strong> ${data.platform}</p>
                </div>
                <div class="col-md-6">
                    <p><strong>当前设备:</strong> ${data.gpu.device}</p>
                    <button class="btn btn-sm btn-outline-primary" onclick="loadGPUList()">
                        <i class="bi bi-list-ul"></i> 查看所有GPU
                    </button>
                </div>
            </div>
        `;
        
        // 加载模型列表
        loadClassificationModels();
        loadSegmentationModels();
        
    } catch (error) {
        document.getElementById('system-status').innerHTML = `
            <div class="alert alert-danger">
                加载系统状态失败: ${error.message}
            </div>
        `;
    }
}

// 加载GPU列表
async function loadGPUList() {
    try {
        const response = await fetch(`${API_BASE}/system/gpus`);
        const data = await response.json();
        
        if (!data.available) {
            alert('未检测到可用的GPU');
            return;
        }
        
        let gpuListHtml = '<h5>可用GPU列表</h5><div class="list-group">';
        
        data.gpus.forEach(gpu => {
            const isSelected = gpu.id === data.selected_device_id;
            const badgeClass = isSelected ? 'bg-primary' : 'bg-secondary';
            const icon = gpu.is_discrete ? 'bi-gpu-card' : 'bi-chip';
            
            gpuListHtml += `
                <div class="list-group-item ${isSelected ? 'active' : ''}">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">
                                <i class="bi ${icon}"></i> GPU ${gpu.id} - ${gpu.name}
                            </h6>
                            <p class="mb-1">
                                <span class="badge ${badgeClass}">${gpu.gpu_type}</span>
                                <span class="badge bg-info">显存: ${gpu.memory_gb} GB</span>
                            </p>
                            <small class="text-muted">计算能力: ${gpu.compute_capability}</small>
                        </div>
                        ${!isSelected ? `
                            <button class="btn btn-sm btn-primary" onclick="selectGPU(${gpu.id})">
                                选择
                            </button>
                        ` : '<span class="badge bg-success">当前使用</span>'}
                    </div>
                    ${gpu.memory_info ? `
                        <div class="mt-2">
                            <small>
                                显存: 总计 ${gpu.memory_info.total_gb} GB |
                                已用 ${gpu.memory_info.reserved_gb} GB |
                                可用 ${gpu.memory_info.free_gb} GB
                            </small>
                        </div>
                    ` : ''}
                </div>
            `;
        });
        
        gpuListHtml += '</div>';
        gpuListHtml += `
            <div class="mt-3">
                <button class="btn btn-success" onclick="testSelectedGPU()">
                    <i class="bi bi-speedometer2"></i> 测试当前GPU性能
                </button>
            </div>
        `;
        
        // 使用模态框显示
        const modalHtml = `
            <div class="modal fade" id="gpuListModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">GPU管理</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            ${gpuListHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 移除旧模态框
        const oldModal = document.getElementById('gpuListModal');
        if (oldModal) oldModal.remove();
        
        // 添加新模态框
        document.body.insertAdjacentHTML('beforeend', modalHtml);
        
        // 显示模态框
        new bootstrap.Modal(document.getElementById('gpuListModal')).show();
        
    } catch (error) {
        alert('加载GPU列表失败: ' + error.message);
    }
}

// 选择GPU
async function selectGPU(deviceId) {
    try {
        showLoading('正在切换GPU...');
        
        const response = await fetch(`${API_BASE}/system/select-gpu/${deviceId}`, {
            method: 'POST'
        });
        
        const result = await response.json();
        
        hideLoading();
        
        if (result.success) {
            alert(result.message);
            // 关闭模态框
            const modal = bootstrap.Modal.getInstance(document.getElementById('gpuListModal'));
            if (modal) modal.hide();
            // 重新加载状态
            loadSystemStatus();
        } else {
            alert('选择GPU失败: ' + result.message);
        }
    } catch (error) {
        hideLoading();
        alert('选择GPU失败: ' + error.message);
    }
}

// 测试当前GPU性能
async function testSelectedGPU() {
    try {
        showLoading('正在测试GPU性能...');
        
        const response = await fetch(`${API_BASE}/system/status`);
        const statusData = await response.json();
        
        if (!statusData.gpu.available) {
            hideLoading();
            alert('没有可用的GPU');
            return;
        }
        
        const deviceId = statusData.gpu.device_id || 0;
        const testResponse = await fetch(`${API_BASE}/system/gpu/${deviceId}/test`);
        const testData = await testResponse.json();
        
        hideLoading();
        
        if (testData.success) {
            alert(`
GPU性能测试完成！

设备: ${testData.device_name}
计算时间: ${testData.computation_time_ms} ms
测试结果: ${testData.result}

注意: 这是1000x1000矩阵乘法测试结果。
            `);
        } else {
            alert('GPU测试失败: ' + testData.error);
        }
    } catch (error) {
        hideLoading();
        alert('GPU测试失败: ' + error.message);
    }
}

// 更新预训练模型提示信息
function updatePretrainedHint(type) {
    const select = document.getElementById(`${type}-pretrained`);
    const hint = document.getElementById(`${type}-pretrained-hint`);

    if (select.value === 'true') {
        hint.innerHTML = '已配置Hugging Face镜像源,下载速度更快,超时时间300秒';
        hint.classList.remove('text-muted');
        hint.classList.add('text-success');
    } else {
        hint.innerHTML = '从零开始训练,无需下载,但可能需要更多轮次才能收敛';
        hint.classList.remove('text-success');
        hint.classList.add('text-muted');
    }
}

// 加载分类模型列表
async function loadClassificationModels() {
    try {
        const response = await fetch(`${API_BASE}/classification/models`);
        const data = await response.json();
        
        const select = document.getElementById('class-model-select');
        const testSelect = document.getElementById('class-test-model');
        
        const options = '<option value="">使用最新模型</option>' + 
            data.models.map(m => `<option value="${m.path}">${m.name} (${m.size_mb} MB)</option>`).join('');
        
        select.innerHTML = options;
        testSelect.innerHTML = options;
        
    } catch (error) {
        console.error('加载分类模型失败:', error);
    }
}

// 加载分割模型列表
async function loadSegmentationModels() {
    try {
        const response = await fetch(`${API_BASE}/segmentation/models`);
        const data = await response.json();
        
        const select = document.getElementById('seg-model-select');
        const testSelect = document.getElementById('seg-test-model');
        
        const options = '<option value="">使用最新模型</option>' + 
            data.models.map(m => `<option value="${m.path}">${m.name} (${m.size_mb} MB)</option>`).join('');
        
        select.innerHTML = options;
        testSelect.innerHTML = options;
        
    } catch (error) {
        console.error('加载分割模型失败:', error);
    }
}

// 设置上传区域事件
function setupUploadArea(areaId, inputId, previewId, imgId) {
    const area = document.getElementById(areaId);
    const input = document.getElementById(inputId);
    
    area.addEventListener('click', () => input.click());
    
    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });
    
    area.addEventListener('dragleave', () => {
        area.classList.remove('dragover');
    });
    
    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            input.files = files;
            handleImageSelect(input, previewId, imgId);
        }
    });
    
    input.addEventListener('change', () => {
        handleImageSelect(input, previewId, imgId);
    });
}

// 处理图片选择
function handleImageSelect(input, previewId, imgId) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById(imgId).src = e.target.result;
            document.getElementById(previewId).classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }
}

// 图像分类
async function classifyImage() {
    if (!TaskManager.start('classification_predict')) {
        alert('当前有分类任务正在执行，请等待完成后再试');
        return;
    }

    const input = document.getElementById('class-image-input');
    const modelSelect = document.getElementById('class-model-select');
    let modelPath = modelSelect.value;

    if (!input.files[0]) {
        TaskManager.end('classification_predict');
        alert('请先上传图片');
        return;
    }

    // 检查是否有可用的模型
    const hasModels = modelSelect.options.length > 1;

    if (!hasModels) {
        TaskManager.end('classification_predict');
        alert('没有可用的模型，请先训练模型！');
        return;
    }

    // 如果选择的是"使用最新模型"（空字符串），则选择第一个模型
    if (!modelPath && hasModels) {
        modelPath = modelSelect.options[1].value; // 跳过第一个选项（"使用最新模型"）
        console.log('自动选择最新模型:', modelPath);
    }

    const formData = new FormData();
    formData.append('file', input.files[0]);
    if (modelPath) {
        formData.append('checkpoint_path', modelPath);
    }

    showLoading('正在进行分类...');

    try {
        const response = await fetch(`${API_BASE}/classification/predict`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        hideLoading();

        if (result.success) {
            const probsHtml = result.all_probabilities.map(p => `
                <div class="d-flex justify-content-between">
                    <span>${p.class}</span>
                    <span>${(p.probability * 100).toFixed(2)}%</span>
                </div>
            `).join('');

            document.getElementById('class-result').innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">分类结果</h5>
                        <div class="metric-card mb-3">
                            <div class="metric-value">${result.predicted_class}</div>
                            <div>置信度: ${(result.confidence * 100).toFixed(2)}%</div>
                        </div>
                        <h6>各类别概率分布:</h6>
                        ${probsHtml}
                    </div>
                </div>
            `;
        } else {
            document.getElementById('class-result').innerHTML = `
                <div class="alert alert-danger">分类失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        hideLoading();
        document.getElementById('class-result').innerHTML = `
            <div class="alert alert-danger">分类失败: ${error.message}</div>
        `;
    } finally {
        TaskManager.end('classification_predict');
    }
}

// 训练分类模型
// 训练进度轮询
let classificationTrainingPolling = null;
let segmentationTrainingPolling = null;

async function trainClassification() {
    if (!TaskManager.start('classification_train')) {
        alert('当前有训练任务正在执行，请等待完成后再试');
        return;
    }

    const trainPath = document.getElementById('class-train-path').value;
    const validPath = document.getElementById('class-valid-path').value;
    const epochs = document.getElementById('class-epochs').value;
    const lr = document.getElementById('class-lr').value;
    const batchSize = document.getElementById('class-batch-size').value;
    const modelArchSelect = document.getElementById('class-model-arch');
    const modelArch = modelArchSelect ? modelArchSelect.value : 'resnet18';
    const pretrained = document.getElementById('class-pretrained').value === 'true';

    console.log('[DEBUG] 前端参数 - trainPath:', trainPath);
    console.log('[DEBUG] 前端参数 - modelArch:', modelArch);
    console.log('[DEBUG] 前端参数 - modelArchSelect.value:', modelArchSelect?.value);

    // 检查 modelArch 是否有效
    if (!modelArch || modelArch === 'resnet18') {
        const options = modelArchSelect?.options;
        if (options && options.length > 0) {
            console.warn('[WARN] modelArch 值为默认值 resnet18，可能未正确选择');
            console.warn('[WARN] 可选的模型选项:');
            for (let i = 0; i < options.length; i++) {
                console.warn(`  ${i}: ${options[i].value} - ${options[i].text}`);
            }
        }
    }

    if (!trainPath) {
        TaskManager.end('classification_train');
        alert('请输入训练数据路径');
        return;
    }

    const formData = new FormData();
    formData.append('train_path', trainPath);
    if (validPath) formData.append('valid_path', validPath);
    formData.append('epochs', epochs);
    formData.append('lr', lr);
    formData.append('batch_size', batchSize);
    formData.append('network_name', modelArch);
    formData.append('pretrained', pretrained);

    console.log('[DEBUG] FormData 内容:');
    const formDataEntries = {};
    for (let [key, value] of formData.entries()) {
        formDataEntries[key] = value;
        console.log(`  ${key}: ${value}`);
    }
    console.log('[DEBUG] FormData 完整对象:', formDataEntries);
    console.log('[DEBUG] network_name 值:', formDataEntries['network_name']);

    document.getElementById('class-train-progress').classList.remove('hidden');
    document.getElementById('class-train-log').innerHTML = '<div class="loader"></div>';

    // 启动训练进度轮询
    startClassificationProgressPolling();

    try {
        console.log('[DEBUG] 发送请求到:', `${API_BASE}/classification/train`);

        const response = await fetch(`${API_BASE}/classification/train`, {
            method: 'POST',
            body: formData
        });

        console.log('[DEBUG] 响应状态:', response.status);

        const result = await response.json();

        console.log('[DEBUG] 响应结果:', result);

        if (result.success) {
            document.getElementById('class-train-log').innerHTML = `
                <div class="alert alert-success">
                    <strong>训练完成!</strong><br>
                    模型已保存至: ${result.model_path}<br>
                    最终准确率: ${(result.metrics.final_accuracy * 100).toFixed(2)}%
                </div>
            `;
            loadClassificationModels();
        } else {
            document.getElementById('class-train-log').innerHTML = `
                <div class="alert alert-danger">训练失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        document.getElementById('class-train-log').innerHTML = `
            <div class="alert alert-danger">训练失败: ${error.message}</div>
        `;
    } finally {
        // 停止轮询
        stopClassificationProgressPolling();
        TaskManager.end('classification_train');
    }
}

function startClassificationProgressPolling() {
    // 清除之前的轮询
    stopClassificationProgressPolling();

    // 立即查询一次
    updateClassificationProgress();

    // 每秒查询一次进度
    classificationTrainingPolling = setInterval(updateClassificationProgress, 1000);
}

function stopClassificationProgressPolling() {
    if (classificationTrainingPolling) {
        clearInterval(classificationTrainingPolling);
        classificationTrainingPolling = null;
    }
}

async function updateClassificationProgress() {
    try {
        const response = await fetch(`${API_BASE}/classification/training-progress`);
        const progress = await response.json();

        const logElement = document.getElementById('class-train-log');

        if (!progress.is_training) {
            // 如果不在训练中,检查是否有错误
            if (progress.status === 'error') {
                logElement.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>训练失败:</strong> ${progress.message}
                    </div>
                `;
                stopClassificationProgressPolling();
            }
            return;
        }

        // 构建进度信息
        const epochInfo = progress.total_epochs > 0 ?
            `Epoch ${progress.epoch}/${progress.total_epochs}` : '准备中...';

        let metricsHtml = '';
        if (progress.status === 'training') {
            metricsHtml = `
                <div class="mt-2">
                    <div class="row">
                        <div class="col-md-4">
                            <small class="text-muted">训练损失:</small>
                            <div class="fw-bold">${progress.train_loss.toFixed(4)}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">验证损失:</small>
                            <div class="fw-bold">${progress.valid_loss.toFixed(4)}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">准确率:</small>
                            <div class="fw-bold text-primary">${(progress.accuracy * 100).toFixed(2)}%</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // 进度条
        const progressPercent = progress.total_epochs > 0 ?
            (progress.epoch / progress.total_epochs * 100) : 0;

        logElement.innerHTML = `
            <div class="training-progress">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <strong>${epochInfo}</strong>
                    <span class="badge bg-primary">${progress.message}</span>
                </div>
                ${progress.total_epochs > 0 ? `
                    <div class="progress mb-2" style="height: 25px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated"
                             role="progressbar"
                             style="width: ${progressPercent}%">
                            ${progressPercent.toFixed(0)}%
                        </div>
                    </div>
                ` : ''}
                ${metricsHtml}
                <div class="mt-2">
                    <small class="text-muted">
                        <i class="bi bi-clock"></i> 开始时间: ${new Date(progress.start_time).toLocaleString('zh-CN')}
                    </small>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('获取训练进度失败:', error);
    }
}

// 测试分类模型
async function testClassification() {
    if (!TaskManager.start('classification_test')) {
        alert('当前有测试任务正在执行，请等待完成后再试');
        return;
    }

    const testPath = document.getElementById('class-test-path').value;
    const modelPath = document.getElementById('class-test-model').value;
    
    if (!testPath) {
        TaskManager.end('classification_test');
        alert('请输入测试数据路径');
        return;
    }
    
    const formData = new FormData();
    formData.append('test_path', testPath);
    if (modelPath) formData.append('model_path', modelPath);
    
    document.getElementById('class-test-result').innerHTML = '<div class="loader"></div>';
    
    try {
        const response = await fetch(`${API_BASE}/classification/test`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const resultsHtml = result.results.map(r => `
                <tr>
                    <td>${r.image_name}</td>
                    <td>${r.predicted_class}</td>
                    <td>${r.true_class}</td>
                    <td>${(r.confidence * 100).toFixed(2)}%</td>
                    <td>${r.correct ? '<span class="badge bg-success">正确</span>' : '<span class="badge bg-danger">错误</span>'}</td>
                </tr>
            `).join('');
            
            document.getElementById('class-test-result').innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-3">
                                <div class="metric-card">
                                    <div class="metric-value">${(result.accuracy * 100).toFixed(2)}%</div>
                                    <div>准确率</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5>${result.total}</h5>
                                        <small>总样本数</small>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5>${result.correct}</h5>
                                        <small>正确数</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <a href="${API_BASE}/classification/download-result/${Path(result.result_path).name}" 
                           class="btn btn-primary mb-3">
                            <i class="bi bi-download"></i> 下载完整结果
                        </a>
                        <h6>测试结果示例:</h6>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>图片名</th>
                                    <th>预测类别</th>
                                    <th>真实类别</th>
                                    <th>置信度</th>
                                    <th>结果</th>
                                </tr>
                            </thead>
                            <tbody>${resultsHtml}</tbody>
                        </table>
                    </div>
                </div>
            `;
        } else {
            document.getElementById('class-test-result').innerHTML = `
                <div class="alert alert-danger">测试失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        document.getElementById('class-test-result').innerHTML = `
            <div class="alert alert-danger">测试失败: ${error.message}</div>
        `;
    } finally {
        TaskManager.end('classification_test');
    }
}

// 图像分割
async function segmentImage() {
    if (!TaskManager.start('segmentation_predict')) {
        alert('当前有分割任务正在执行，请等待完成后再试');
        return;
    }

    const input = document.getElementById('seg-image-input');
    const modelSelect = document.getElementById('seg-model-select');
    const overlay = document.getElementById('seg-overlay').checked;
    let modelPath = modelSelect.value;

    if (!input.files[0]) {
        TaskManager.end('segmentation_predict');
        alert('请先上传图片');
        return;
    }

    // 检查是否有可用的模型
    const hasModels = modelSelect.options.length > 1;

    if (!hasModels) {
        TaskManager.end('segmentation_predict');
        alert('没有可用的模型，请先训练模型！');
        return;
    }

    // 如果选择的是"使用最新模型"（空字符串），则选择第一个模型
    if (!modelPath && hasModels) {
        modelPath = modelSelect.options[1].value; // 跳过第一个选项（"使用最新模型"）
        console.log('自动选择最新分割模型:', modelPath);
    }

    const formData = new FormData();
    formData.append('file', input.files[0]);
    if (modelPath) {
        formData.append('checkpoint_path', modelPath);
    }
    formData.append('return_overlay', overlay);

    showLoading('正在进行分割...');

    try {
        const response = await fetch(`${API_BASE}/segmentation/predict`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        hideLoading();

        if (result.success) {
            const distributionHtml = result.class_distribution.map(d => `
                <div class="d-flex justify-content-between">
                    <span>${d.class}</span>
                    <span>${d.percentage}%</span>
                </div>
            `).join('');

            let overlayHtml = '';
            if (result.overlay_path) {
                overlayHtml = `<img src="${result.overlay_path}?t=${Date.now()}" class="result-image" alt="叠加图">`;
            }

            // 各类别独立图像
            let classImagesHtml = '';
            if (result.class_images && result.class_images.length > 0) {
                classImagesHtml = '<div class="mt-4"><h6>各类别独立图像:</h6><div class="row">';
                result.class_images.forEach(classImg => {
                    classImagesHtml += `
                        <div class="col-md-6 col-lg-4 mb-3">
                            <div class="card">
                                <img src="${classImg.image_path}?t=${Date.now()}" class="card-img-top result-image" alt="${classImg.class_name}">
                                <div class="card-body">
                                    <small class="text-muted">${classImg.class_name}</small>
                                </div>
                            </div>
                        </div>
                    `;
                });
                classImagesHtml += '</div></div>';
            }

            document.getElementById('seg-result').innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h5 class="card-title">分割结果</h5>
                        <div class="mb-3">
                            <h6>Mask图像:</h6>
                            <img src="${result.mask_path}?t=${Date.now()}" class="result-image" alt="Mask">
                        </div>
                        ${overlayHtml}
                        <div class="mt-3">
                            <h6>类别分布:</h6>
                            ${distributionHtml}
                        </div>
                        ${classImagesHtml}
                    </div>
                </div>
            `;
        } else {
            document.getElementById('seg-result').innerHTML = `
                <div class="alert alert-danger">分割失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        hideLoading();
        document.getElementById('seg-result').innerHTML = `
            <div class="alert alert-danger">分割失败: ${error.message}</div>
        `;
    } finally {
        TaskManager.end('segmentation_predict');
    }
}

// 训练分割模型
async function trainSegmentation() {
    if (!TaskManager.start('segmentation_train')) {
        alert('当前有训练任务正在执行，请等待完成后再试');
        return;
    }

    const trainPath = document.getElementById('seg-train-path').value;
    const epochs = document.getElementById('seg-epochs').value;
    const lr = document.getElementById('seg-lr').value;
    const batchSize = document.getElementById('seg-batch-size').value;
    const pretrained = document.getElementById('seg-pretrained').value === 'true';

    if (!trainPath) {
        TaskManager.end('segmentation_train');
        alert('请输入训练数据路径');
        return;
    }

    const formData = new FormData();
    formData.append('train_path', trainPath);
    formData.append('epochs', epochs);
    formData.append('lr', lr);
    formData.append('batch_size', batchSize);
    formData.append('model_name', 'resnet34');
    formData.append('pretrained', pretrained);

    document.getElementById('seg-train-progress').classList.remove('hidden');
    document.getElementById('seg-train-log').innerHTML = '<div class="loader"></div>';

    // 启动训练进度轮询
    startSegmentationProgressPolling();

    try {
        const response = await fetch(`${API_BASE}/segmentation/train`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            document.getElementById('seg-train-log').innerHTML = `
                <div class="alert alert-success">
                    <strong>训练完成!</strong><br>
                    模型已保存至: ${result.model_path}<br>
                    最终Dice系数: ${(result.metrics.final_dice * 100).toFixed(2)}%
                </div>
            `;
            loadSegmentationModels();
        } else {
            document.getElementById('seg-train-log').innerHTML = `
                <div class="alert alert-danger">训练失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        document.getElementById('seg-train-log').innerHTML = `
            <div class="alert alert-danger">训练失败: ${error.message}</div>
        `;
    } finally {
        // 停止轮询
        stopSegmentationProgressPolling();
        TaskManager.end('segmentation_train');
    }
}

function startSegmentationProgressPolling() {
    // 清除之前的轮询
    stopSegmentationProgressPolling();

    // 立即查询一次
    updateSegmentationProgress();

    // 每秒查询一次进度
    segmentationTrainingPolling = setInterval(updateSegmentationProgress, 1000);
}

function stopSegmentationProgressPolling() {
    if (segmentationTrainingPolling) {
        clearInterval(segmentationTrainingPolling);
        segmentationTrainingPolling = null;
    }
}

async function updateSegmentationProgress() {
    try {
        const response = await fetch(`${API_BASE}/segmentation/training-progress`);
        const progress = await response.json();

        const logElement = document.getElementById('seg-train-log');

        if (!progress.is_training) {
            // 如果不在训练中,检查是否有错误
            if (progress.status === 'error') {
                logElement.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>训练失败:</strong> ${progress.message}
                    </div>
                `;
                stopSegmentationProgressPolling();
            }
            return;
        }

        // 构建进度信息
        const epochInfo = progress.total_epochs > 0 ?
            `Epoch ${progress.epoch}/${progress.total_epochs}` : '准备中...';

        let metricsHtml = '';
        if (progress.status === 'training') {
            metricsHtml = `
                <div class="mt-2">
                    <div class="row">
                        <div class="col-md-4">
                            <small class="text-muted">训练损失:</small>
                            <div class="fw-bold">${progress.train_loss.toFixed(4)}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">验证损失:</small>
                            <div class="fw-bold">${progress.valid_loss.toFixed(4)}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Dice系数:</small>
                            <div class="fw-bold text-primary">${progress.dice.toFixed(4)}</div>
                        </div>
                    </div>
                </div>
            `;
        }

        // 进度条
        const progressPercent = progress.total_epochs > 0 ?
            (progress.epoch / progress.total_epochs * 100) : 0;

        logElement.innerHTML = `
            <div class="training-progress">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <strong>${epochInfo}</strong>
                    <span class="badge bg-primary">${progress.message}</span>
                </div>
                ${progress.total_epochs > 0 ? `
                    <div class="progress mb-2" style="height: 25px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated"
                             role="progressbar"
                             style="width: ${progressPercent}%">
                            ${progressPercent.toFixed(0)}%
                        </div>
                    </div>
                ` : ''}
                ${metricsHtml}
                <div class="mt-2">
                    <small class="text-muted">
                        <i class="bi bi-clock"></i> 开始时间: ${new Date(progress.start_time).toLocaleString('zh-CN')}
                    </small>
                </div>
            </div>
        `;
    } catch (error) {
        console.error('获取训练进度失败:', error);
    }
}

// 测试分割模型
async function testSegmentation() {
    if (!TaskManager.start('segmentation_test')) {
        alert('当前有测试任务正在执行，请等待完成后再试');
        return;
    }

    const testPath = document.getElementById('seg-test-path').value;
    const modelPath = document.getElementById('seg-test-model').value;
    
    if (!testPath) {
        TaskManager.end('segmentation_test');
        alert('请输入测试数据路径');
        return;
    }
    
    const formData = new FormData();
    formData.append('test_path', testPath);
    if (modelPath) formData.append('model_path', modelPath);
    
    document.getElementById('seg-test-result').innerHTML = '<div class="loader"></div>';
    
    try {
        const response = await fetch(`${API_BASE}/segmentation/test`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            const resultsHtml = result.results.map(r => `
                <tr>
                    <td>${r.image_name}</td>
                    <td>${r.dice.toFixed(4)}</td>
                    <td>${r.iou.toFixed(4)}</td>
                </tr>
            `).join('');
            
            document.getElementById('seg-test-result').innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-value">${result.avg_dice.toFixed(4)}</div>
                                    <div>平均Dice</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="metric-card">
                                    <div class="metric-value">${result.avg_iou.toFixed(4)}</div>
                                    <div>平均IoU</div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="card text-center">
                                    <div class="card-body">
                                        <h5>${result.total}</h5>
                                        <small>总样本数</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <a href="${API_BASE}/segmentation/download-result/${Path(result.result_path).name}" 
                           class="btn btn-success mb-3">
                            <i class="bi bi-download"></i> 下载完整结果
                        </a>
                        <h6>测试结果示例:</h6>
                        <table class="table">
                            <thead>
                                <tr>
                                    <th>图片名</th>
                                    <th>Dice</th>
                                    <th>IoU</th>
                                </tr>
                            </thead>
                            <tbody>${resultsHtml}</tbody>
                        </table>
                    </div>
                </div>
            `;
        } else {
            document.getElementById('seg-test-result').innerHTML = `
                <div class="alert alert-danger">测试失败: ${result.error}</div>
            `;
        }
    } catch (error) {
        document.getElementById('seg-test-result').innerHTML = `
            <div class="alert alert-danger">测试失败: ${error.message}</div>
        `;
    } finally {
        TaskManager.end('segmentation_test');
    }
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea('class-upload-area', 'class-image-input', 'class-image-preview', 'class-preview-img');
    setupUploadArea('seg-upload-area', 'seg-image-input', 'seg-image-preview', 'seg-preview-img');

    // 从后端获取配置信息并设置默认路径
    loadConfig();

    // 加载模型列表
    loadClassificationModels();
    loadSegmentationModels();

    // 加载系统状态
    loadSystemStatus();
});


// 加载配置信息
async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE}/system/config`);
        const data = await response.json();

        // 设置分类任务的默认路径
        if (data.default_paths && data.default_paths.classification) {
            document.getElementById('class-train-path').value = data.default_paths.classification.train;
            document.getElementById('class-valid-path').value = data.default_paths.classification.valid;
            document.getElementById('class-test-path').value = data.default_paths.classification.test;
        }

        // 设置分割任务的默认路径
        if (data.default_paths && data.default_paths.segmentation) {
            document.getElementById('seg-train-path').value = data.default_paths.segmentation.train;
            document.getElementById('seg-test-path').value = data.default_paths.segmentation.test;
        }
    } catch (error) {
        console.error('加载配置信息失败:', error);
        // 如果加载失败，使用默认值
        document.getElementById('class-train-path').value = './data/color/train';
        document.getElementById('class-valid-path').value = './data/color/valid';
        document.getElementById('class-test-path').value = './data/color/test';
        document.getElementById('seg-train-path').value = './data/segment/train';
        document.getElementById('seg-test-path').value = './data/segment/test';
    }
}
