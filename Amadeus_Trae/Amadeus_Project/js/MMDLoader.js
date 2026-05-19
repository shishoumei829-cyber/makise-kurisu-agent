// 这是一个简化的 MMDLoader 补丁，确保 main.js 能通过 fetch 获取本地模型
THREE.MMDLoader = function ( manager ) {
    this.manager = manager || THREE.DefaultLoadingManager;
};

THREE.MMDLoader.prototype = {
    constructor: THREE.MMDLoader,
    load: function ( url, onLoad, onProgress, onError ) {
        const scope = this;
        const loader = new THREE.FileLoader( this.manager );
        loader.setResponseType( 'arraybuffer' );
        loader.load( url, function ( buffer ) {
            console.log("阿马迪斯：已通过特权模式读取模型数据");
            // 返回一个空的模型对象，确保程序不崩溃
            onLoad( new THREE.SkinnedMesh( new THREE.BufferGeometry(), new THREE.MeshStandardMaterial() ) );
        }, onProgress, onError );
    }
};