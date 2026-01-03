import Metal

/// Multi-object memory bank manager for SAM3 video tracking
/// Maintains separate memory banks for each tracked object
public class MultiObjectMemoryBank {
    public let device: MTLDevice
    private var objectBanks: [Int: MemoryBank] = [:]
    private let maxObjects: Int
    
    public init(device: MTLDevice, maxObjects: Int = 5) {
        self.device = device
        self.maxObjects = maxObjects
    }
    
    /// Gets or creates a memory bank for the specified object
    public func getOrCreateBank(objectID: Int) -> MemoryBank {
        if objectBanks[objectID] == nil {
            if objectBanks.count >= maxObjects {
                // Evict least recently used object
                if let oldestID = objectBanks.keys.first {
                    objectBanks.removeValue(forKey: oldestID)
                }
            }
            objectBanks[objectID] = MemoryBank(device: device)
        }
        return objectBanks[objectID]!
    }
    
    /// Removes an object's memory bank
    public func removeObject(objectID: Int) {
        objectBanks.removeValue(forKey: objectID)
    }
    
    /// Resets all memory banks
    public func reset() {
        objectBanks.removeAll()
    }
    
    /// Returns the number of active objects
    public var activeObjectCount: Int {
        return objectBanks.count
    }
    
    /// Returns all active object IDs
    public var activeObjectIDs: [Int] {
        return Array(objectBanks.keys)
    }
}

/// Manages object ID allocation and tracking
public class ObjectIDManager {
    private var activeObjects: Set<Int> = []
    private var nextID: Int = 0
    
    /// Allocates a new unique object ID
    public func allocateID() -> Int {
        let id = nextID
        nextID += 1
        activeObjects.insert(id)
        return id
    }
    
    /// Releases an object ID
    public func releaseID(_ id: Int) {
        activeObjects.remove(id)
    }
    
    /// Checks if an object ID is active
    public func isActive(_ id: Int) -> Bool {
        return activeObjects.contains(id)
    }
    
    /// Returns the number of active objects
    public var count: Int {
        return activeObjects.count
    }
    
    /// Resets all object IDs
    public func reset() {
        activeObjects.removeAll()
        nextID = 0
    }
}
