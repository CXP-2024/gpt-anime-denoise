package com.animedenoise.core

import java.util.concurrent.ForkJoinPool
import java.util.concurrent.RecursiveAction
import kotlin.math.max
import kotlin.math.min

/**
 * Small work-stealing helper used by every multi-threaded stage of the library.
 *
 * All heavy loops are expressed as "do this for index @i@ in @0 until count@", where the
 * index ranges over independent work items - an image row, an FFT column, a channel.
 * [forEach] spreads those items over a [ForkJoinPool] so a phone can use its big cores,
 * and blocks the calling thread until the whole range is finished.
 *
 * The pool is capped at [MAX_THREADS] because the work items here are memory bound: past
 * roughly eight threads the memory bandwidth is saturated and the little cores only add
 * contention.
 *
 * Tests and benchmarks can pin the library to one thread by setting [threads] to 1, which
 * turns [forEach] into a plain sequential loop.
 */
internal object Parallel {

    /** Number of worker threads [forEach] will use right now. */
    val threadCount: Int get() = currentThreads

    /** Upper bound on worker threads, whatever the device reports. */
    private const val MAX_THREADS = 8

    /** One thread per core, capped at [MAX_THREADS]. */
    private val defaultThreads: Int = max(1, min(Runtime.getRuntime().availableProcessors(), MAX_THREADS))

    @Volatile
    private var currentThreads: Int = defaultThreads

    @Volatile
    private var pool: ForkJoinPool = ForkJoinPool(currentThreads)

    /**
     * Number of threads used by [forEach].
     *
     * 1 (or less) makes every parallel loop sequential; larger values rebuild the pool.
     * Intended for benchmarking and for tests that need deterministic scheduling.
     */
    var threads: Int
        get() = currentThreads
        set(value) {
            val v = max(1, min(value, 64))
            if (v == currentThreads) return
            val old = pool
            currentThreads = v
            pool = ForkJoinPool(v)
            old.shutdown()
        }

    /**
     * Whether the parallel loops are currently running sequentially.
     *
     * Callers use it to skip work that only pays off when the result is shared between
     * threads (for example keeping a scratch line per thread).
     */
    val isSequential: Boolean get() = currentThreads <= 1

    /**
     * Runs @action(0)@ through @action(count - 1)@ on the worker pool, blocking until every
     * item has completed.
     *
     * The range is split recursively until a leaf holds at most @count / (8 * threads)@
     * items (but at least one), which balances uneven per-item cost well. The calling
     * thread takes part in the work.
     *
     * @throws RuntimeException the first exception thrown by [action], rethrown here
     */
    /**
     * Runs [action] over contiguous ranges covering [0, count).
     *
     * Preferred over [forEach] in hot loops. [forEach] invokes its lambda once per element, and on
     * ART that call is not reliably inlined, so a per-row callback became the dominant cost of the
     * L0 solve (measured 10x slower on device than on a desktop JVM for identical work). Ranges let
     * the caller write a plain loop the optimiser can keep in registers.
     */
    fun forEachRange(count: Int, action: (from: Int, to: Int) -> Unit) {
        if (count <= 0) return
        val t = currentThreads
        if (t <= 1 || count == 1) {
            action(0, count)
            return
        }
        // One task per thread, minimum one element each; far fewer objects than the per-element
        // path and each leaf does a long sequential run.
        val parts = min(t, count)
        val base = count / parts
        val extra = count % parts
        if (parts == 1) {
            action(0, count)
            return
        }
        val tasks = ArrayList<RangeAction>(parts)
        var start = 0
        for (p in 0 until parts) {
            val len = base + if (p < extra) 1 else 0
            tasks.add(RangeAction(start, start + len, action))
            start += len
        }
        // invokeAll submits to the common pool and blocks until every task finishes.
        java.util.concurrent.ForkJoinTask.invokeAll(tasks)
    }

    private class RangeAction(
        private val from: Int,
        private val to: Int,
        private val action: (Int, Int) -> Unit
    ) : RecursiveAction() {
        override fun compute() = action(from, to)
    }

    fun forEach(count: Int, action: (Int) -> Unit) {
        if (count <= 0) return
        val t = currentThreads
        if (t <= 1 || count == 1) {
            for (i in 0 until count) action(i)
            return
        }
        val chunk = max(1, count / (t * 8))
        pool.invoke(RangeTask(0, count, chunk, action))
    }

    /** Leaf work item carrying the per-index action. */
    private class RangeTask(
        private val lo: Int,
        private val hi: Int,
        private val chunk: Int,
        private val action: (Int) -> Unit
    ) : RecursiveAction() {
        override fun compute() {
            if (hi - lo <= chunk) {
                for (i in lo until hi) action(i)
                return
            }
            val mid = (lo + hi) ushr 1
            invokeAll(RangeTask(lo, mid, chunk, action), RangeTask(mid, hi, chunk, action))
        }
    }
}
