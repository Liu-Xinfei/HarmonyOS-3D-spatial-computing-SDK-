/**
 * HarmonySpatial SDK — AR 空间锚点与平面检测
 * 封装鸿蒙 AR Engine / 分布式空间感知接口
 * 包含：SpatialAnchor、PlaneDetector、AnchorManager
 * @module ar
 */
import { Vector3 } from '../math/Vector3'
import { Quaternion } from '../math/Quaternion'
import { Matrix4 } from '../math/Matrix4'
import { Plane } from '../geometry/Geometry'

interface PlanePoint2D {
  x: number
  z: number
}

export interface PlaneExtent {
  width:  number
  height: number
}

export interface AnchorPoseEntry {
  pose:  SpatialPose
  state: AnchorState
}

export interface AnchorJSON {
  id:         string
  state:      AnchorState
  position:   number[]
  rotation:   number[]
  timestamp:  number
  confidence: number
}

// ─────────────────────────────────────────────────────────────
// 类型定义
// ─────────────────────────────────────────────────────────────
export type AnchorState =
  | 'initializing'
  | 'tracking'
  | 'paused'
  | 'stopped'
  | 'error'

export type PlaneOrientation = 'horizontal_up' | 'horizontal_down' | 'vertical' | 'arbitrary'

export interface SpatialPose {
  position: Vector3
  rotation: Quaternion
  timestamp: number        // 毫秒，相对设备启动时间
  confidence: number       // 0..1
}

// ─────────────────────────────────────────────────────────────
// SpatialAnchor — 现实世界中的固定点
// ─────────────────────────────────────────────────────────────
export class SpatialAnchor {
  readonly id: string
  state: AnchorState = 'initializing'
  pose: SpatialPose

  private _onUpdate?: (anchor: SpatialAnchor) => void
  private _onStateChange?: (state: AnchorState) => void

  constructor(
    id: string,
    initialPose: SpatialPose
  ) {
    this.id   = id
    this.pose = { position: initialPose.position, rotation: initialPose.rotation, timestamp: initialPose.timestamp, confidence: initialPose.confidence }
  }

  onUpdate(cb: (anchor: SpatialAnchor) => void): SpatialAnchor {
    this._onUpdate = cb
    return this
  }

  onStateChange(cb: (state: AnchorState) => void): SpatialAnchor {
    this._onStateChange = cb
    return this
  }

  /** 由 AR 引擎调用，更新姿态数据 */
  _update(pose: SpatialPose, state: AnchorState) {
    const prevState = this.state
    this.pose  = { position: pose.position, rotation: pose.rotation, timestamp: pose.timestamp, confidence: pose.confidence }
    this.state = state
    this._onUpdate?.(this)
    if (state !== prevState) this._onStateChange?.(state)
  }

  /** 获取当前世界变换矩阵 */
  getWorldMatrix(): Matrix4 {
    return Matrix4.compose(
      this.pose.position,
      this.pose.rotation,
      Vector3.one()
    )
  }

  /** 将世界坐标转换为锚点局部坐标 */
  worldToAnchorLocal(worldPos: Vector3): Vector3 {
    return this.getWorldMatrix().inverse().transformPoint(worldPos)
  }

  isTracking(): boolean { return this.state === 'tracking' }

  toJSON(): AnchorJSON {
    const pos = this.pose.position
    const rot = this.pose.rotation
    const timestamp = this.pose.timestamp
    const confidence = this.pose.confidence
    const result: AnchorJSON = {
      id:         this.id,
      state:      this.state,
      position:   pos.toArray(),
      rotation:   [rot.x, rot.y, rot.z, rot.w],
      timestamp,
      confidence,
    }
    return result
  }

  static fromJSON(data: AnchorJSON): SpatialAnchor {
    return new SpatialAnchor(data.id, {
      position:   Vector3.fromArray(data.position),
      rotation:   new Quaternion(data.rotation[0] as number, data.rotation[1] as number, data.rotation[2] as number, data.rotation[3] as number),
      timestamp:  data.timestamp,
      confidence: data.confidence,
    })
  }
}

// ─────────────────────────────────────────────────────────────
// DetectedPlane — AR 检测到的物理平面
// ─────────────────────────────────────────────────────────────
export class DetectedPlane {
  readonly id: string
  orientation: PlaneOrientation
  center: Vector3
  normal: Vector3
  extent: PlaneExtent          // 米
  polygon: Vector3[]                          // 平面轮廓顶点（世界坐标）
  confidence: number

  get plane(): Plane {
    return Plane.fromNormalPoint(this.normal, this.center)
  }

  constructor(
    id: string,
    orientation: PlaneOrientation,
    center: Vector3,
    normal: Vector3,
    extent: PlaneExtent,
    polygon: Vector3[],
    confidence: number = 1
  ) {
    this.id          = id
    this.orientation = orientation
    this.center      = center
    this.normal      = normal.normalize()
    this.extent      = extent
    this.polygon     = polygon
    this.confidence  = confidence
  }

  /** 测试点是否在平面多边形内（射线法） */
  containsPoint2D(worldPoint: Vector3): boolean {
    // 投影到平面局部 XZ 平面
    const right   = Vector3.up().cross(this.normal).normalize()
    const forward = this.normal.cross(right).normalize()
    const local   = worldPoint.sub(this.center)
    const px = local.dot(right)
    const pz = local.dot(forward)

    let inside = false
    const poly: PlanePoint2D[] = this.polygon.map(v => {
      const l = v.sub(this.center)
      const pt: PlanePoint2D = { x: l.dot(right), z: l.dot(forward) }
      return pt
    })

    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const pi = poly[i], pj = poly[j]
      if (
        (pi.z > pz) !== (pj.z > pz) &&
        px < (pj.x - pi.x) * (pz - pi.z) / (pj.z - pi.z) + pi.x
      ) {
        inside = !inside
      }
    }
    return inside
  }

  area(): number {
    // 计算多边形面积（Shoelace 公式，投影到平面）
    const right   = Vector3.up().cross(this.normal).normalize()
    const forward = this.normal.cross(right).normalize()
    let area = 0
    const n = this.polygon.length
    for (let i = 0; i < n; i++) {
      const a = this.polygon[i]
      const b = this.polygon[(i + 1) % n]
      const ax = a.sub(this.center).dot(right),  az = a.sub(this.center).dot(forward)
      const bx = b.sub(this.center).dot(right),  bz = b.sub(this.center).dot(forward)
      area += ax * bz - bx * az
    }
    return Math.abs(area) / 2
  }
}

// ─────────────────────────────────────────────────────────────
// PlaneDetector — 管理平面检测
// ─────────────────────────────────────────────────────────────
export interface PlaneDetectorOptions {
  minArea?: number              // 最小面积（m²），过滤噪声
  orientations?: PlaneOrientation[]
  updateInterval?: number       // 毫秒
}

export class PlaneDetector {
  readonly planes: Map<string, DetectedPlane> = new Map()
  private _onAddedCb:   ((p: DetectedPlane) => void) | null = null
  private _onUpdatedCb: ((p: DetectedPlane) => void) | null = null
  private _onRemovedCb: ((id: string) => void) | null = null
  private _options: Required<PlaneDetectorOptions>
  private _running: boolean = false

  constructor(options: PlaneDetectorOptions = {} as PlaneDetectorOptions) {
    this._options = {
      minArea:        options.minArea        ?? 0.1,
      orientations:   options.orientations   ?? ['horizontal_up', 'vertical'],
      updateInterval: options.updateInterval ?? 100,
    }
  }

  onPlaneAdded(cb: (p: DetectedPlane) => void): PlaneDetector {
    this._onAddedCb = cb; return this
  }
  onPlaneUpdated(cb: (p: DetectedPlane) => void): PlaneDetector {
    this._onUpdatedCb = cb; return this
  }
  onPlaneRemoved(cb: (id: string) => void): PlaneDetector {
    this._onRemovedCb = cb; return this
  }

  /**
   * 启动检测（对接鸿蒙 AR Engine）
   * 实际项目中替换为 ohos.ar.arEngine.startPlaneDetection()
   */
  start(): void {
    if (this._running) return
    this._running = true
    // 插入鸿蒙 AR Engine 回调绑定点：
    // arEngine.on('planeAdded',   (data) => this._onNativePlaneAdded(data))
    // arEngine.on('planeUpdated', (data) => this._onNativePlaneUpdated(data))
    // arEngine.on('planeRemoved', (id)   => this._onNativePlaneRemoved(id))
  }

  stop(): void {
    this._running = false
    // arEngine.off(...)
  }

  /** 注入模拟平面（测试/预览用） */
  _injectPlane(plane: DetectedPlane): void {
    if (plane.area() < this._options.minArea) return
    if (!this._options.orientations.includes(plane.orientation)) return

    const isNew = !this.planes.has(plane.id)
    this.planes.set(plane.id, plane)
    if (isNew) {
      if (this._onAddedCb !== null) this._onAddedCb(plane)
    } else {
      if (this._onUpdatedCb !== null) this._onUpdatedCb(plane)
    }
  }

  _removePlane(id: string): void {
    if (!this.planes.has(id)) return
    this.planes.delete(id)
    if (this._onRemovedCb !== null) this._onRemovedCb(id)
  }

  /** 查找距离点最近的水平面 */
  findNearestHorizontal(point: Vector3): DetectedPlane | null {
    let best: DetectedPlane | null = null
    let bestDist = Infinity
    for (const p of this.planes.values()) {
      if (p.orientation !== 'horizontal_up') continue
      const d = point.distanceTo(p.center)
      if (d < bestDist) { bestDist = d; best = p }
    }
    return best
  }
}

// ─────────────────────────────────────────────────────────────
// AnchorManager — 锚点生命周期管理
// ─────────────────────────────────────────────────────────────
export class AnchorManager {
  private _anchors: Map<string, SpatialAnchor> = new Map()
  private _counter: number = 0

  get anchors(): ReadonlyMap<string, SpatialAnchor> { return this._anchors }

  /**
   * 在指定位置创建锚点
   */
  createAnchor(
    position: Vector3,
    rotation: Quaternion = Quaternion.identity()
  ): SpatialAnchor {
    const id = `anchor_${++this._counter}`
    const anchor = new SpatialAnchor(id, {
      position,
      rotation,
      timestamp:  Date.now(),
      confidence: 1,
    })
    this._anchors.set(id, anchor)
    return anchor
  }

  /**
   * 在检测到的平面上创建锚点
   */
  createAnchorOnPlane(
    plane: DetectedPlane,
    localOffset: Vector3 = Vector3.zero()
  ): SpatialAnchor {
    const right   = Vector3.up().cross(plane.normal).normalize()
    const forward = plane.normal.cross(right).normalize()
    const position = plane.center
      .add(right.scale(localOffset.x))
      .add(forward.scale(localOffset.z))
      .add(plane.normal.scale(localOffset.y))

    const rotation = Quaternion.fromToRotation(Vector3.up(), plane.normal)
    return this.createAnchor(position, rotation)
  }

  removeAnchor(id: string): boolean {
    return this._anchors.delete(id)
  }

  getAnchor(id: string): SpatialAnchor | undefined {
    return this._anchors.get(id)
  }

  /** 更新所有锚点（AR Engine 帧回调中调用） */
  updateAll(poseMap: Map<string, AnchorPoseEntry>): void {
    poseMap.forEach((data, id) => {
      this._anchors.get(id)?._update(data.pose, data.state)
    })
  }

  /** 导出持久化数据，用于跨会话恢复 */
  exportAnchors(): object[] {
    return Array.from(this._anchors.values()).map(a => a.toJSON())
  }

  clear(): void { this._anchors.clear() }
}

// ─────────────────────────────────────────────────────────────
// SpatialHitTest — 命中测试（射线投射到真实世界）
// ─────────────────────────────────────────────────────────────
export interface HitResult {
  type:     'plane' | 'anchor' | 'mesh'
  distance: number
  position: Vector3
  normal:   Vector3
  plane?:   DetectedPlane
  anchor?:  SpatialAnchor
}

export class SpatialHitTest {
  /**
   * 屏幕点命中测试
   * @param screenX  0..1 归一化屏幕 X
   * @param screenY  0..1 归一化屏幕 Y
   * @param planes   候选平面列表
   * @param maxDist  最大检测距离（米）
   */
  static hitTestPlanes(
    screenX: number,
    screenY: number,
    planes: DetectedPlane[],
    cameraPos: Vector3,
    cameraFwd: Vector3,
    cameraUp: Vector3,
    fovY: number,
    aspect: number,
    maxDist: number = 10
  ): HitResult | null {
    // 简化射线构建（生产中从 Camera.screenToRay 获取）
    const right = cameraFwd.cross(cameraUp).normalize()
    const ndcX = screenX * 2 - 1
    const ndcY = 1 - screenY * 2
    const halfH = Math.tan(fovY / 2)
    const dir = cameraFwd
      .add(right.scale(ndcX * halfH * aspect))
      .add(cameraUp.scale(ndcY * halfH))
      .normalize()

    let best: HitResult | null = null
    for (const plane of planes) {
      const pl = plane.plane
      const denom = pl.normal.dot(dir)
      if (Math.abs(denom) < 1e-6) continue
      const t = -(pl.normal.dot(cameraPos) + pl.d) / denom
      if (t <= 0 || t > maxDist) continue
      const hitPos = cameraPos.add(dir.scale(t))
      if (!plane.containsPoint2D(hitPos)) continue
      if (!best || t < best.distance) {
        const hit: HitResult = {
          type:     'plane',
          distance: t,
          position: hitPos,
          normal:   plane.normal.clone(),
          plane,
        }
        best = hit
      }
    }
    return best
  }
}
